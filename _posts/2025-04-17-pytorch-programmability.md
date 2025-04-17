---
title: "Enhancing PyTorch Backend Programmability with User-Defined Context
  Hints"
date: 2025-04-17
categories:
  - optimization
tags:
  - pytorch
  - AI
  - optimization
classes: wide
---

PyTorch and torch.compile
=========================

PyTorch is a popular open-source machine learning library known for its
flexibility and ease of use, particularly in research and development
settings. One of the core challenges with PyTorch has been its default
mode of operation, known as eager execution. While eager execution
allows for intuitive and immediate computation, it can hinder
performance optimization because operations are executed as they are
called, without a bigger view of the computation graph and calling
frontend every time specific code is executed, causing severe host
bottlenecks at times.

To address this, PyTorch introduced torch.compile, a graph-based
optimization extension. torch.compile transforms PyTorch\'s eager
execution mode into a graph-based execution mode, where the entire
computational graph is analyzed and optimized as a whole before
execution. When such optimized code is called again, it will be reused.
This approach brings several advantages:

-   **Enhanced Computation Performance:** By viewing the entire graph,
    torch.compile can apply a range of optimizations that are not
    possible with eager execution. These include operator fusion, memory
    optimization, and scheduling improvements.

-   **Reduced Host Activity:** Optimizing and executing whole graphs
    instead of single operations causes significant reduction in host
    activity.

Overall, torch.compile bridges the gap between the ease of use provided
by eager execution and the high performance required for production
environments, making it a crucial tool for developers and researchers
working with complex neural network models. But there are limitations
that even torch.compile itself cannot solve.

Leap forward
============

Today, optimizing workload execution is crucial for achieving high
performance and operating cost reduction. One innovative approach to
achieving this is by allowing PyTorch users to provide additional
context or \"hints\" to the torch.compile backend. This capability can
guide the backend in making more informed decisions about execution
strategies, ultimately leading to better performance and resource
utilization and sometimes being the only way to execute big,
memory-intensive workloads at all. This article delves into the
technical details of user-defined context hints in the Gaudi stack,
focusing on a specific use case of Flash Attention by providing a
practical example.

High-Level Concept
==================

The proposed feature aims to allow end-users to inject additional
context into the backend operations of the PyTorch framework. This
capability is designed to be highly flexible, enabling users to attach
custom information to specific operations as they see fit. By doing so,
the backend can utilize this contextual information to refine
optimization strategies and execution schemes, aligning more closely
with the user\'s expectations and needs.

There are many advantages of this feature, offering a significant boost
in the versatility of programmability. Users can influence various
aspects of execution, such as selecting computational streams,
determining the order of operations, suggesting slicing schemes, and
grouping operations into bundles. For example, in scheduling, users
could have the option to specify their preferred execution order,
whether it be Breadth-First Search (BFS), Depth-First Search (DFS), or
even a more tailored execution sequence. These custom execution orders
can meet specific user requirements, offering a granular level of
control over the execution flow.

While these user-provided \"hints\" are designed as optional, they add
an extra layer of potential for backend optimizations. However, it is
crucial to note that any backend retains the discretion to disregard
these hints if they are incompatible with the system\'s functional or
performance requirements. This ensures that while user input is valued
and considered, the overall system integrity and efficiency are not
compromised. This forward-thinking approach not only enriches the user
experience but also pushes the boundaries of what can be achieved with
customized backend optimizations in PyTorch.

Specific Use Case: Flash Attention Operation
============================================

To illustrate this concept, let\'s consider the flash attention
operation. This operation may contain multiple potential parallel
execution paths and as such, it can be executed in either a Depth-First
Search (DFS) or Breadth-First Search (BFS) manner. Each execution
strategy has its pros and cons:

-   **BFS Execution:** This approach allows for more parallel paths to
    be executed simultaneously, facilitating better parallelization
    across execution engines. However, it requires more accelerator
    memory to accommodate multiple parallel slices at once.

-   **DFS Execution:** This method allocates memory for a single
    parallel path at a time, reusing intermediate tensors for subsequent
    paths. While it may sacrifice some of the raw performance, it
    significantly reduces memory usage, making it more suitable for
    accelerators with limited memory.

Given these trade-offs, the optimal execution strategy depends on the
specific constraints and priorities of the user\'s workload.
Unfortunately, current big limitation of backend compilers is that they
lack contextual information that would allow them to choose this
strategy on their own as they have no knowledge about user expectation.

### *And this is where aforementioned Context Hints come in.*

For below example needs, let\'s assume that we are limited by
accelerator memory usage, and we might want to trade-off a bit of
performance if we can save significant amount of this accelerator
memory. For each experiment we will show the performance of the
operation (time required to finish executing forward and backward parts
taken together on the accelerator, read from the gathered hardware
traces) as well as its' memory usage (computed as maximum peak memory
used during first run minus persistent memory used for inputs and
outputs to the test, using built-in functionality of Habana PyTorch
integration that provides extensive memory analytics to the users). Test
is ran on following shapes of the matrices:

  | Q shape | (8, 32, 2048, 128) |
  | K shape | (8, 32, 2048, 128) |
  | V shape | (8, 32, 2048, 128) |

Let\'s dive into a concrete implementation example of SDPA part of flash
attention, following basic implementation supports both mode without
slicing (to be precise, having single slice) as well as having multiple
slices. For simplicity, we will include only forward part of the
implementation, as there is no additional mechanisms used for backward
apart from those that will be explained here for the forward.


```python
def sdpa_fwd(ctx, q, k, v, is_causal, with_slice):
    """
    1. using retain tensor
    2. if slice enabled, using default size 1
    """
    # no slice on batch heads dimensions
    batch_heads = 1
    if with_slice:
        batch_heads = q.shape[0] * q.shape[1]

    scale = 1 / math.sqrt(q.size(-1))

    Q = torch.flatten(q, end_dim=1)
    K = torch.flatten(k, end_dim=1)
    V = torch.flatten(v, end_dim=1)

    Qi = torch.tensor_split(Q, batch_heads)
    Ki = torch.tensor_split(K, batch_heads)
    Vi = torch.tensor_split(V, batch_heads)

    retain_exp_slices = []
    retain_max_slices = []
    output_slices = []
    for slice in range(batch_heads):
        current_Qi = Qi[slice]
        current_Ki = Ki[slice]
        current_Vi = Vi[slice]

        Si = torch.matmul(current_Qi, current_Ki.transpose(-2, -1))

        if is_causal:
            Pi, retain_exp, retain_max = torch.ops.hpu.scaled_triangular_softmax_retain(Si, scale)

            retain_exp_slices.append(retain_exp.unsqueeze(0))
            retain_max_slices.append(retain_max.unsqueeze(0))
        else:
            Si = torch.mul(Si, scale)
            Pi = F.softmax(Si, dim=-1)

        output_slices.append(torch.matmul(Pi, current_Vi))

    retain_exp = None
    retain_max = None
    if is_causal:
        retain_exp = torch.cat(retain_exp_slices)
        retain_max = torch.cat(retain_max_slices)

    return (
        torch.cat(output_slices).reshape((q.shape[0], q.shape[1], v.shape[2], v.shape[3])),
        retain_exp,
        retain_max,
    )

class PySDPA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False, with_slice=False):

        out, retain_exp, retain_max = sdpa_fwd(ctx, q, k, v, is_causal, with_slice)

        ctx.save_for_backward(q, k, v, out, retain_exp, retain_max)

        ctx.is_causal = is_causal
        ctx.with_slice = with_slice

        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, retain_exp, retain_max = ctx.saved_tensors
        dq, dk, dv = sdpa_bwd(dout, q, k, v, out, ctx.is_causal, retain_exp, retain_max, ctx.with_slice)
        return dq, dk, dv, None, None, None
```


We will first run it while making sure there is single slice
(with\_slice=False), which would show the performance and memory usage
for initial simple implementation user would create for new operation.
We start in eager mode with this experiment, and results are as
following:

  | Peak memory used \[MB\]   | Time to execute \[ms\]  |
  | ------------------------- | ----------------------- |
  | 8842                      | 23.5                    |

We have no comparison as of performance yet, but we can already see that
for this simple operation we take need huge amount of memory due to big,
unsliced matrix created by Q\*K gemm operation.

Now, let\'s add some parallelization to this by enabling the slicing
(with\_slice=True) which effectively creates 128 parallel chunks.
Expectation from the user would be, that now when we explicitly created
these parallel paths and execute them one by one, we will see
significant memory reuse:

  | Peak memory used \[MB\]   | Time to execute \[ms\]  |
  | ------------------------- | ----------------------- |
  | 424                       | 262                     |

As we can see, user would be right and PyTorch framework along with
Habana integration is able to reuse the memory to some extent, but we
can do more as we will see shortly. Also, what did just happen with the
performance? It's order of magnitude worse, and the reason is that by
slicing we have just created many much smaller tasks and in eager mode
for each such tasks we need to go through the whole stack and back,
causing severe host bottleneck. This is not the solution for the
original memory issue we want to use, even though memory usage is indeed
lower.

This is why next example is going to use torch.compile so that backend
can see whole code in single graph, effectively being able to do more
optimizations and should remove the host bottleneck visibly for our
sliced code. There is no difference in the implementation of the
operation, everything will be just wrapped by torch.compile extension
using HPU backend. Result of the experiment is:

  | Peak memory used \[MB\]   | Time to execute \[ms\]  |
  | ------------------------- | ----------------------- |
  | 2043                      | 21                      |

What happened here? Performance we've got is the best as of our current
experiments but we again see quite a lot of memory being used there. The
reason is that backend compiler wanted to maximize the performance at
the cost of memory usage optimizations and it scheduled the work using
BFS scheme. Why it actually chose BFS? As we said earlier, we created
over a hundred of small slices of work and it is much easier for the
compiler to provide maximum performance by parallelizing them over all
engines. There is no default way to tell compiler it should optimize for
memory instead. Using Context Hints, you can tell it to execute it using
DFS (actually it is \"strict\" scheme we use here, which means we will
execute in the same order as it would in eager, which will effectively be
DFS) as it physically allows for memory reuse we wanted.

To use this feature, you need to wrap the functions inside
autograd.Function to use Higher Order Op which is going to pass the
hints from the user to the backend, here is how the definition looks
like now (do note that the internal implementation of the SDPA did not
change at all, and changes revolve mostly around autograd.Function using
HOO to call this implementation). We will again focus on forward:

```python
class PySDPAHinted(torch.autograd.Function):
    """
    1. using retain tensor
    2. if slice enabled, using default size 1
    """

    @staticmethod
    def forward(ctx, q, k, v, is_causal=False, with_slice=True):
        def forward_hinted(ctx, q, k, v, is_causal, with_slice, hint):
            # using default slice size 1
            batch_heads = q.shape[0] * q.shape[1]

            scale = 1 / math.sqrt(q.size(-1))

            Q = torch.flatten(q, end_dim=1)
            K = torch.flatten(k, end_dim=1)
            V = torch.flatten(v, end_dim=1)

            Qi = torch.tensor_split(Q, batch_heads)
            Ki = torch.tensor_split(K, batch_heads)
            Vi = torch.tensor_split(V, batch_heads)

            retain_exp_slices = []
            retain_max_slices = []
            output_slices = []
            for slice in range(batch_heads):
                current_Qi = Qi[slice]
                current_Ki = Ki[slice]
                current_Vi = Vi[slice]

                Si = torch.matmul(current_Qi, current_Ki.transpose(-2, -1))

                if is_causal:
                    Pi, retain_exp, retain_max = torch.ops.hpu.scaled_triangular_softmax_retain(Si, scale)

                    retain_exp_slices.append(retain_exp.unsqueeze(0))
                    retain_max_slices.append(retain_max.unsqueeze(0))
                else:
                    Si = torch.mul(Si, scale)
                    Pi = F.softmax(Si, dim=-1)

                output_slices.append(torch.matmul(Pi, current_Vi))

            # workaround for the issue:
            #  "HigherOrderOperator body's output must consist of tensors only"
            outputs = [torch.cat(output_slices).reshape((q.shape[0], q.shape[1], v.shape[2], v.shape[3]))]

            retain_exp = None
            retain_max = None
            if is_causal:
                outputs.append(torch.cat(retain_exp_slices))
                outputs.append(torch.cat(retain_max_slices))

            return tuple(outputs)

        if is_causal:
            out, retain_exp, retain_max = hinted_context(
                forward_hinted,
                ctx,
                q,
                k,
                v,
                is_causal,
                with_slice,
                hint='{"order": "strict", "group_id": 0}',
            )
            ctx.save_for_backward(q, k, v, out, retain_exp, retain_max)
        else:
            (out,) = hinted_context(
                forward_hinted,
                ctx,
                q,
                k,
                v,
                is_causal,
                with_slice,
                hint='{"order": "strict", "group_id": 0}',
            )
            ctx.save_for_backward(q, k, v, out)

        ctx.with_slice = with_slice
        ctx.is_causal = is_causal

        return out
```

Results now are as following:

  | Peak memory used \[MB\]   | Time to execute \[ms\]  |
  | ------------------------- | ----------------------- |
  | 55                        | 22.7                    |

We can see that this mechanism allowed user to tell the backend compiler
that DFS execution is the expected one, even if performance might be a
bit lower.

Let's summarize all the results and the steps we took:

  | Scenario              | Peak memory used \[MB\]   | Time to execute \[ms\]  |
  | --------------------- | ------------------------- | ----------------------- |
  | Naïve eager           | 8842                      | 23.5                    |
  | Sliced eager          | 424                       | 262                     |
  | Sliced graph          | 2043                      | 21                      |
  | Hinted sliced graph   | 55                        | 22.7                    |


-   First, we used naïve eagerly executed implementation which tried to
    execute whole operation at once, including huge GEMM inside.

-   Second, we sliced the inner loop to break the operations into many
    smaller chunks, still executing in eager, that allowed for some
    memory reuse but broke the performance.

-   Third, we wrapped our sliced implementation with torch.compile,
    which allowed us to remove host bottleneck, improving the
    performance a lot but memory savings were lost a bit there again due
    to default compiler strategy which is to optimize for raw
    performance, which in turn causes it to favor BFS scheme for such
    many tasks, effectively trying to allocate all slices at once again.

-   Finally, we added hints to our wrapped implementation, by which we
    told compiler to schedule work using different scheme of DFS, which
    might not be optimal for performance, but might allow for
    significant memory saving. And this is exactly what we see, no other
    configuration allowed us to get to 55 MB of peak memory required
    (**\~37x** improvement over not hinted graph and **\~160x**
    improvement over original naïve eager code) while being not that
    much slower than the fastest option. The tradeoff here could be
    managed even more by carefully choosing the number of slices used.

Conclusion
==========

User-defined context hints provide a powerful mechanism for enhancing
the programmability and flexibility of neural network execution. By
allowing end-users to specify their preferences and constraints,
backends can optimize execution strategies more effectively, leading to
better performance and resource utilization. The implementation of such
a system in the PyTorch framework, as demonstrated on Gaudi stack in
this article, showcases the potential benefits and practical
applications of this approach.

By incorporating these techniques, developers can achieve more efficient
and tailored execution of complex operations, ultimately advancing the
capabilities of neural network frameworks and hardware accelerators.

Final words from author
==========

This blog post of mine was initially written like a year ago, I think, but it
was never actually published, it wasn't checked by technical writer as well so
I hope it is not too bad. As the code was open sourced, I decided to publish it
myself ^^

Current implementation was opensourced and you can
check it here:

[PySDPA OP implementation](https://github.com/HabanaAI/gaudi-pytorch-bridge/blob/v1.20.0/python_packages/habana_frameworks/torch/hpex/kernels/PySDPA.py)

[Test running this and comparing with other scenarios](https://github.com/HabanaAI/gaudi-pytorch-bridge/blob/v1.20.0/tests/pytest_working/any_mode/test_hpu_multiple_sdpa_impls.py)

Also, special thanks to Zhang Wuxun for his support when we worked on this feature.

Thanks for reading.