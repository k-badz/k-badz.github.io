---
title: "Low overhead profiling"
date: 2025-04-16
categories:
  - optimization
tags:
  - lop
  - profiling
  - optimization
---

Hello everyone. In this post we will go through the story and cool stuff about [LOP](https://github.com/k-badz/low-overhead-profiler) project repository.

## Motivation
Imagine you have a project that needs to do some stuff and it needs to do it very fast. Fast to the point where each chunk of work can take up to few microseconds. On top of that you have multiple threads that talk to each other, each of them does completely different stuff, it gets the work from its' queue, does something and then passes to the other threads to process further. Kind of pipelining done on the CPU host side. If you disturb the system, even very lightly, some phenomena (like queues being overloaded, causing latencies in the system) might disappear because by introducing overhead, you might have just created an execution bubble that gives more time for others to work. And believe me - when you have like 3 microseconds budget for work chunk - it is very easy to disturb the system.

Now, you need to optimize it. At first you will surely use VTune, Perf, Callgrind, or whatever similar tool that will give you some initial information, like specific functions taking too much time, poor communication between threads (to some extent) and so on. But at some point you might hit a wall and ask yourself:

"What the hell is actually hapenning in there? I have no more obvious hotspots, but system still seems to not perform as I would like it to. And when I profile it, it even behaves differently!"

At this point, you might start adding various time points to your code yourself, to see how much **exactly** specific parts are taking. And then you discover, that again it does not behave as it would in production after you did your simple manual profiling changes. Maybe you used std::chrono, or some other fancy mechanism that takes anything between 500-5000ns to get a single measurement. You get my meaning - single sample can take more time than your whole work chunk which might consist of multiple functionalities..

I have been there, and this is how Low Overhead Profiler was born.


## Idea
General idea behind it was simple. Do whatever is needed to get a profiler that can gather events with overhead measurable in tens of cycles, not in microseconds. Additional requirement - it needs to be simple to use, no dependencies, no Gigabytes downloads, no shady software installations, no fancy drivers to enable performance counters and so on, as quite often what you get if you go that route - is a week of debugging the profiler itself, sometimes without any resolution. 

Sounds ambitious, so let's break this idea into reality. What do we need to record that something happened at some time?

1. We need to ask the CPU to tell us current time. We have stuff for that, RDTSC, RDTSCP, RDPMC.
2. We need **zero** synchronization between threads. So we will give each a separate buffer for events.
3. If we have separate "anything" for each thread, we need TLS. I actually did custom one.
4. Each time we gather an event, we cannot do any fancy checks other than necessary. So no "typical" thread_local/static/whatever singleton here folks.
5. We need to postprocess the data. We will do it offline, after profiler is stopped.

Now, we will go through some of these points and explain stuff and also cleanup some controversies.

## Design choices discussions
I made some very unusal design choices and we will explain them in discussion-like manner.

### To RDTSC or not to RDTSC
I'm using RDTSC and my project assumes you are working on modern, strongly-ordered-memory x64 architecture processor on modern OS. Why it is important? Let's discuss.

**The TSC (time stamp counter) on your CPU might not be properly synchronized between cores, so it can completely change anytime your thread migrate.**

This is argument that comes quite often on forums and internets. Once upon a time, indeed there were such issues. But if your processor is not like 20 years old, then you can forget about that point. Even OSes mechanisms related to scheduling and high resolution stuff, are using TSC. When you boot your CPU, any reasonable OS will issue instructions to synchronize those. There might be theoretically an issue with multiple NUMA nodes being used though, so be wary (although I did not see this in practice). To summarize - in theory it could be a problem, but on proper OS like Windows or Ubuntu and on a modern x64 CPU this is only theoretical issue.

**Okay smart-ass, but TSC measures CPU cycles, and frequency changes all the time.**

Common argument, but TSC unit has its own constant frequency that is set during boot and is not dependent on current CPU frequency or even CPU power level (in specifically that regard it is even better than PMC!). It usually is very close to nominal frequency of your CPU. No worries, TSC is the stablest thing you can use to measure real time flowing on your CPU.

**Yeah, but RDTSC do not measure how much my process was >actually< being executed on CPU, contrary to my lovely terribly-slow OS-provided alternatives, why should I care about the time apart from it?**

Because the structure of your workload might affect how often your process gets pre-empted. Because world does not stop when your process stops working. Because in the end, customer looks how much time he spent on a coffee before your workload finished. Because you might use shitty OS scheduler and maybe it is time to use different one (both OS and scheduler). There are many reasons, and due to above I (usually) consider alive-process-only-time-measurements not that helpful. It is also very enlightening to see how often your process is not being processed by any CPU, and you can see that if you use RDTSC. It is a real issue that might come up if your CPU has too much work on it during the workload and the most precious part of it might wait simply too much for its' time quant.

**Wait, whoa, let's go back a bit, what does it mean that x64 is strongly-ordered?**

It means that store requests to your CPU are always done (or, to say it more properly - their results are globally visible) in sync with the original assembly listing. In other words, it means that, if you want some memory-storing parts of your code to happen in exactly the same order (as observed by e.g. other threads) - just make sure compiler does not shuffle those around. This is common mistake done by people who start using RDTSC to overlook it - they put two RDTSC close to each other and then compiler swaps them behind the scenes.... 

I'm using assembly directly in parts where it could theoretically matter, and compiler barriers between events, so I don't care.

**Compiler barriers? But it might break some optimization opportunities for my original code**

Yup, this is where you are perfectly right. If you spotted that, great job! You cannot put those events all over the place, every line or two, because this is exactly what will happen. I recommend to add them only on beginning and end of the functions (use Scoped profilers for that ideally) as in big projects you are going to have multiple compilation units that call each other all the time, and as practice shows - noone uses whole program optimization for various reasons - which means there would be no optimizations anyway there. But even with this, what I usually do is first I add those traces all over the place to find interesting regions, and then I leave only those interesting regions to reduce that kind of overhead.

**Why no RDPMC? It is faster**

Yes, but it also requires you to enable reading performance counters which usually requires kernel mode module for that and it might be even not possible to enable it for security reasons in your enviroment. So we stick to RDTSC.

**Okay, but I've heard it is better to use RDTSCP, or maybe RDTSC+LFENCE or maybe CPUID+WHATEVER as they give you better measurement accuracy due to their side effect of flushing the CPU pipeline and so on.**

If I were measuring exact cycles in a CPU microbenchmark, this is exactly what I would do. But we are not doing microbenchmark here. Each such synchronization will induce additional constant overhead of tens of cycles, for each event you gather. If you do that, you destabilize the initial system execution, effectively reducing your **overall** measurement accuracy. Funny, right? So we stick to RDTSC, it will indeed shuffle around due to out of order execution, but knowing that x64 is strongly ordered and we use compiler barriers, it will do so in one direction only, and not really that much. No worries.

### Why no typical singleton? Or just static global variable?
Singleton pattern is cool, but every time you call something like getInstance() for it, you will do a check whether it was already initialized. Global static variable is also cool as we wouldn't have those checks, it even allows - at least in Linux - you to use the same profiler instance betwen different libraries as linux loader will merge same symbols from different libraries so you can get automatically merged traces (I mean, events from different components merged into single trace) if you use that all over your stack. But that's illegal.. I mean, it just works, but you really shouldn't rely on anything to merge your global static variable between different libraries.

So let's get all the best stuff, with no cons at all. It is called inline global variable. It can be legally used in multiple translation units and symbol will be merged (although I didn't test the runtime merging :x), and has no runtime checks (as it is expected to be initialized at this point). Oh yeah, this and structured bindings are only two things that are necessary to make C++17 standard useful. I love it.

### Custom TLS? Really?
Yeah, I know, usually you don't need to care about it as built-in TLS implementation is no longer that shitty these days. But shared objects sometimes (due to not being main binary, but one that could be loaded into virtually anything) use very slow implementation of it (I actually traced it through assembly in GDB and I ended in some function called "slow" or "slowest" in TLS implementation, after many checks in the between, yuck).

What my "custom TLS" is doing, it relies on the fact that both Windows and Ubuntu can get the thread id (or pointer to thread structure like pthread_self in case of Linux) very quickly (one assembly instruction to be exact), and they both use at most 16 bits of that value, as I tested by creating thousands of threads. So what we will do? We will create static lookup table that can contain like 65k entries (pointers) and those 16bits will be an offset in that table. So each time we emit an event and want to get our TLS, we just check if TLS pointer in that static table is nonzero. This is fast enough, just single memory load that will be cached anyway. The biggest overhead of TLS initialization will happen only on first event of that thread. And 65k entries * 8 bytes of pointers is like 512KB of RAM needed - c'mon, these days just moving your mouse over empty desktop requires more than that. Sounds like a deal. 

As of portability - yes, this is not portable. While I'm using some segment registers offsets that for Windows seem to didn't change since beginning of time, it might not be the case for Linux distributions. If you encounter any issues related to it, check MacroTLSCheck definition in profiler_asm.asm (Windows) and profiler_asm.cpp (Linux). If you think you need more bits used in your case, you need to change CUSTOM_TLS_SIZE define value in profiler.cpp and tweak the aforementioned MacroTLSCheck definition.

### No security checks? Bruh...
Yup, each thread will allocate RAM space for like 4 million events per thread. Feel free to lower it if you don't need that much and you are short on RAM - just change LOP_BUFFER_SIZE define value in profiler.cpp.

I don't think you will ever need more than that on the other hand as it would result in traces being so big that they couldn't be loaded in chrome tracing nor perfetto anyway.

If you're up for a challenge, you might recreate my original version of the idea where I used ring buffers so overflow wouldn't ever happen, in worst case you would just lose some events. But you will pay additional two instructions penalty (ADD+AND) per each event if you do that. Original code was even better - it had additional thread that actually actively flushing all the thread rings to some single big buffer or even disk to not lose any event. It didn't even require any thread synchronization (due to proper usage of ring buffer and assumption on being strongly ordered architecture) so it also was very fast. But IMHO that system was too complicated and I just dumped it into a trash as I prefer simplicity `¯\_(ツ)_/¯`

### Can we get lower with the overhead? Where that 8 nanoseconds come from? 
Yes, and no, frankly. 

No, because 8 nanoseconds is time required to execute the single RDTSC itself (over 20 cycles, yup, it's heavy) and such good result happens when other instructions hides into your usual system latencies thanks to some partial out of order execution that is happening behind the scenes. This is why I say it has 8-12 nanoseconds overhead, because this is the range that might happen in real life. Proper way to measure it, is to run your full system with and without it, measure the total time difference and divide by the number of events you gathered.

Yes, because I provided special events, like the endbegin and immediate, that can create actually two events with single RDTSC used. So in specific cases you can, kinda, get lower.

### Wait wait, before you go, how can I enable MASM in Visual Studio?
1. Right-click the project
2. Build Dependencies
3. Build Customizations
4. Tick "masm"
5. Right-click profiler_asm.asm
6. Properties
7. Set "Item Type" to "Microsoft Macro Assembler"

## Wrapping up
I hope you enjoy this project, have fun :)
