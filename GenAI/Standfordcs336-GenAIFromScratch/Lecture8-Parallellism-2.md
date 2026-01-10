This is week 2 of the systems lecture, where
we try to leverage the most out of the hardware we have to make models train faster.
And last week, we talked about parallelism within a single GPU,
and this week, we're talking about parallelism across multiple GPUs. So this is a picture you should have in your head.
So we have a bunch of nodes. These are basically computers that each
have a number of GPUs-- usually 8. And within each GPU, there's a bunch of streaming multiprocessors, or SMs, which actually do the work.
And you see that in green here are essentially the memory and the communications. Within each SM, you have a very small L1 cache.
On a GPU, you have a high-bandwidth memory, HBM, which is bigger. And then you have these links that connect the different GPUs.
So the way to think about it is that compute has to happen within the SM on these ALUs.
And a compute needs inputs and needs to write outputs. And generally, the inputs and outputs can be relatively far.
If you're lucky, they're on the L1 cache. If you're slightly less unlucky, they're in HBM.
And now this week, we're talking about multi-GPU and multi-node training where the data that you might need
might be across on another GPU. So the name of the game is, how do you structure all your computation to avoid data transfer
bottlenecks? Because we want to, remember, keep the arithmetic intensity
high. We want to saturate our GPUs, make them hum along. And generally, data transfer is going to be a lot slower,
so we have to-- that's going to be the bottleneck. So last week we saw a bunch of different techniques
to try to do that within a GPU, including fusion and tiling. So the idea basically is that instead of reading and writing
from HBM, you can load into L1 cache, or I guess a shared memory, which
is using the same type of-- has the same speed, and just work there
on your local scratchpad and then write out to HBM only judiciously.
And this week, we started looking at the communication across GPUs and nodes where we have to replicate and shard our models and parameters
and optimize our states, and there, the way we do that will determine the cost.
So here's a-- I'm taking a little bit of liberty to put everything in one hierarchy,
you can think, from small/fast to big/slow. So the smallest and fastest is on a single node, single GPU,
you have L1 cache that's extremely fast, but very small. And then you have HBM on a single GPU.
And then between GPUs on the same node, we have NVLink,
and then finally, we have NVSwitch. And of course, this is all in the NVIDIA ecosystem.
So the idea is that many of the core concepts of minimizing data
transfer are really the same, but now the mechanics are a bit different because L1 behaves differently
than these NV switches. So this lecture is going to be mostly
about concretizing the concepts from the lecture in code. There's going to be a few new things,
but Tatsu did an excellent job of giving you an overview of all the different types of parallelism.
I'm going to try to anchor it in the code so we can more deeply understand what's going on.
And then we're going to have-- I'm going to refer to this standard out file
here, which is the output of running this lecture. There were some minor issues I'll spare you
of where if you have multiprocessing, then this framework doesn't quite work.
So this lecture has two parts. In part 1, we're going to look at the building blocks--
collective operations, which we discussed last time, how this is implemented in NCCL and PyTorch,
and then we're going to do some benchmarking. And then in part 2, we're going to look at actually
distributed training, data, tensor, and pipeline parallelism.
OK. So let's start with collective operations. So collective operations are these primitives
that are used generally for distributed programming, and collective means that you have many nodes.
These are actually quite old, from at least the '80s and the parallel programming literature.
And generally, they provide a better abstraction than trying to manage the point-to-point communication
yourself. So these are really tried-and-true primitives that
have stood the test of time. So a bit of terminology. So world size refers to essentially the number
of devices-- for example, 4. And the rank, confusingly, if you're used to linear algebra,
actually just refers to divide. So we have rank 0, rank 1, rank 2, and rank 3 if you have four devices.
OK, so the collective operations are as follows.
So starting from broadcast, the idea is you have t0 on one of the ranks,
and you just want to put it on all the other ranks-- or all ranks. So that's very straightforward.
Scatter is similar, but you have four values, and you want to put each of the values on different ranks.
So each of the ranks get different values, not the same value.
Gather is the inverse of scatter where you have each rank having a different value, and then you bring them all together
onto one rank. Reduce is the same as gather, except for instead
of concatenating, you add them. all_gather is the same as gather,
except for you just do it for all the destinations. Gather was just rank 0, or rank 1
or rank 2, or any individual rank. all_gather as you do it for all of them.
And then finally, reduce_scatter-- I couldn't find a good picture of this, so I'm reusing the one from last time,
is like reduce where you take a bunch of different values
and you add them or perform other commutative operation on them and put it on one rank.
But like scatter, you're going to be putting different pieces of the vector
or tensor on different ranks. And remember that all_reduce is equivalent to reduce
plus all_gather. So the way to remember this terminology
is as follows because it can get confusing which one's all_gather, which one's reduce_scatter,
is that reduce just means you're performing some associative and commutative operation like sum or min or max
or average. Broadcast-scatter is the inverse of gather.
And all it just means, a destination is all devices.
So hopefully this is a review from last time.
So, is there any questions before I move on?
Since we're going to build on these primitives, so it's useful if everyone understands.
OK. So now let's see how this is actually implemented
in starting with the hardware. So here's classically what hardware for GPUs looks like.
So this is in the home. You have a computer, I guess.
And you have your CPUs. And generally, you have your GPUs
on one node that communicate via PCI-E bus.
And if you have to go communicate between different nodes, then this
is all connected to ethernet. So this is typically how machines were built. If you buy a GPU and you're for gaming or something,
this is probably what your setup looks like. As we'll see, this is suboptimal because there's
a lot of overhead. When the data needs to get shipped from GPU to GPU,
it has to go through the kernel, get copied into buffers, and then go through this transport over Ethernet,
and that introduces a lot of overhead. So what has happened in modern times
with scientific computing and deep learning is that if you know that you're going
to just string a bunch of GPUs together and do something together, then we're just going to hook the GPUs up directly,
basically. So in the NVIDIA ecosystem, we have NVLink that directly connects the GPUs, therefore,
bypassing the CPU-- you don't need to go through the kernel of the host machine.
And even across nodes, we can connect the GPUs directly via NVSwitch.
So therefore, we're bypassing ethernet. And because ethernet was developed a long time ago, clearly not for these type of applications.
So NVSwitch and NVLink skip all of that and just optimize directly for the type of workloads
that we're interested in. So, if you look at H100s, each node has--
or sorry, each GPU has 18 NVLinks generation 4 coming out.
So that gives you a total bandwidth of 900 gigabytes. If you compare to these, it's certainly
a lot faster than PCI-E, and it's certainly way faster than ethernet.
And in comparison, if you think about the cost of just going
from the SM to reading from high-bandwidth memory, that's still quite a bit faster by a factor of 4 or so.
And of course, these numbers are constantly changing. With the new Blackwells, this number is two or three times
more, I believe. OK. Yeah? So the PC, [INAUDIBLE] through the [INAUDIBLE] CPU
and then another GPU? Or it's a direct link between the GPUs? So the question is for the PCI-E,
how does the data get transferred? I think it has to still go through the CPU.
Was there another question?
And the PCI-E was-- I mean, it's developed for things like-- other things are connected to it as well,
like your sound card or your SSD or hard drive. So it's not really-- it's a general purpose bus
for communication of devices. Yeah? NVLink has a connection with the CPU?
Yeah. So the question is, NVLink also connects to the CPU? We're going to see a bit later how--
I think maybe just in this slide, how things are connected. Yeah. So you still need to talk to your CPU, of course.
Yeah. OK, so there's this command you can run,
and this produces some output, which allows you to see how the GPUs are actually connected.
So I ran this on our cluster. There is 8 GPUs--
I guess you won't be able to get 8 GPUs, but I guess if you could, this is what it would look like.
And you see that between every pair of GPUs, there's NV18 connecting.
There's also these network cards and other things.
OK.
Oh yeah. So then network cards are basically what gives you the PCI-E connection and the CPUs.
OK, so that's the hardware.
So how do you use the hardware? So NVIDIA has spent a lot of time developing really
good software on top of their, I guess, really good hardware. And there's a collective communication library
by NVIDIA called NCCL. And this essentially translates the collective operations,
which we looked at before, like all_reduce, into low-level packets that need to be sent between GPUs.
So this library actually does a lot of work because it allows the programmer just to operate that level of,
I need this tensor to appear on all the machines, and it just happens.
So just a little bit of what happens is when you configure, set up NCCL, you bring up
a bunch of devices. And there's some communication that
happens to figure out the topology of the hardware. It optimizes the path between the GPUs. And then when you actually call these collective communication
operations, it then launches CUDA kernels to send and receive data. OK, so that's the NCCL.
That's provided as a library. But NCCL is still a bit too low level to us
because most of what we're doing is in Python. So PyTorch has this torch.distributed library
which essentially provides a clean interface for these collective operations. Now from the comfort of your PyTorch program,
you can just write all_gather into tensor on a tensor and it will appear on all the different ranks.
It also has this nice useful feature
that it supports multiple backends for different hardware. So in particular, NCCL, remember, was for GPU,
but you can also run collective operations-- remember, this is not GPU-specific, it's just for any set of devices.
So you can also do it for CPU using this backend called gloo. So if you're debugging stuff on your laptop for your assignment,
for example, you can use gloo and still be able to run things without even a GPU.
So anyway, that's another advantage of having these high-level primitives, is that they're much more portable than
having to-- only having something that's very GPU-specific.
Of course, the performance is going to really depend on the hardware, but at least logically, you
can make sure your code runs. Distributed also supports other high-level things
like FSDP, which Tatsu talked about last lecture, but we're not going to use in this class
because in the spirit of developing things from scratch, that's just what we're going to do.
OK, so let's look at some examples of how torch.distributed collective operations work.
So there's this utility function I wrote, which you can take a look at it in the code if you want,
which takes a function and just runs this-- basically, it's a wrapper around Python multiprocessing
where it just runs four processes that execute this function.
So when you're in this function, you should think about it as there's actually world_size number of processes running this identical function
where the rank indexes from 0, 1, all the way to world_size minus 1.
So right now, I'm stepping through just one of the ranks because lectures are not parallel.
And so generally what you do is you-- the first thing, the process needs to initialize itself.
And you essentially-- they need to find each other because you're a multi-processor,
you're running a lot of processes, they need to connect to a single host so that they can figure--
know that each other exist. So note that this is not where all of the data goes.
The data goes through, NCCL, but this is just for coordination. And since we a GPU, we can use NCCL,
otherwise you would use gloo. OK, so after you set up--
so now we're going to do some stuff. There's this useful function called barrier which basically
waits for all the processes in your process group to get to this point.
Remember, everything is running asynchronously. And in some cases, you just want to have a synchronization point,
so barrier does that. The reason I put it here is actually for trivial reasons
because I want all these print statements to be grouped together, but there's other reasons
why you might want to use barrier as we'll get to later. So I'm going to, for each of these groups,
construct a tensor. So the tensor is 0, 1, 2, 3 plus the rank.
So I'm going to print out for each rank, before the all_reduce, what does it look like?
OK, so here's what it looks like. Can people read that in the back?
Yes? OK, good. All right, so on rank 0, it's 0, 1, 2, 3.
For rank 1, 1, 2, 3, 4, and so on. And notice that because it's async, the order is-- it's just out of order in whatever order it happens to print.
So each rank has a different tensor.
And then you all_reduce. So all_reduce, you pass in that tensor.
You say I want to sum it. In this case, I'm not going to do async, but you can do async which is useful for overlapping
communication and computation. And then afterwards, what happens after all_reduce,
as advertised, basically for the first component, you add them up, you get 6.
This, you get 10, 14, and 18. So after all_reduce, basically, this tensor
gets overwritten with the corresponding sum.
So it's very, very nice and simple to use. OK.
So, let's do reduce_scatter. So reduce_scatter, I'm going to create
an input, which has dimension world_size, in which case,
this is 4. And I'm going to allocate an output because reduce_scatter
is not going to operate in place, this is just going to be a scalar. So before the reduce_scatter, this is what it looks like.
I have my input as before. Output happens to be 0's, but it could be any value
since I didn't initialize it. And then after the reduce_scatter,
when I pass in the input and the output, and I'm going to sum, then I get--
essentially what happens is that for the first component, I sum, and that goes on rank 0; for the second component, I sum,
and it goes on rank 1, and so on. So as you notice, it is producing the same operation
as all_reduce except for the output is scattered across all the different ranks.
So now let's do all_gather.
So I'm going to just directly use the output reduce_scatter,
which is this, as the input. And then I'm going to allocate an empty array for the output.
And then so before the all_gather, the input is this,
and the output, I guess, are just arbitrary values.
And after I do the all_gather, what happens
is I get all these tensors to show up on all the devices.
So this is just a also an example.
Hopefully now you're very convinced that reduce_scatter plus all_gather is just all_reduce because I computed
exactly the same quantity as I did for all_reduce.
OK. Questions? Is this clear?
Yeah? in reduce_scatter, are we keeping track of which index goes to which GPU?
So the question is, in reduced-scatter, do you keep track of which index goes to which GPU?
So by convention, the dimensionality has to be,
then, basically the-- I mean it could be a general tensor, but one of the dimensions is the world_size,
and it just infers that basically what you want to do is
the output is the-- let's say the-- sorry.
The input has to be basically world_size, and then it knows that basically,
the corresponding computations go to each of the outputs.
Yeah, you have to be a bit careful with making sure the dimensionality align. So going through this with small examples can be helpful.
Is there another question?
So finally, we're now in this process that's running,
and when you're done, you just clean up. OK.
So so far, we've talked about these collective operations, a bit about how they're implemented
in PyTorch, and its-- NCCL and then PyTorch. Let's do a bit of benchmarking in the spirit of what we did
in assignment-- or the first lecture-- or rather, the second lecture.
We're going to focus on one node for now. So let's do all_reduce.
So I'm going to have this tensor of 100 million elements
and a world_size of 4. So I'm going to just allocate a tensor.
And generally, as I think-- as you hopefully can appreciate now, that when you benchmark,
you have to really be careful to clean your palette in some sense.
In this case, I'm going to warm up, basically run the operation once, and then synchronize and do barrier.
Some of this is, I think, probably a bit defensive, but just to be safe so that all the kernels get loaded and whatever needs
to be computed gets computed. And then I'm going to start the clock,
all_reduce, and then synchronize again, and stop the clock.
So, now I can look at how long that took.
So if I scroll down here-- I guess this is not that informal.
I should have printed in microseconds, probably. It was, I guess, very quick.
Some number of seconds. And now, let's measure the bandwidth, which
is the number of gigabytes that were actually transferred in aggregate per second.
So the way we do that is we have to think about what actually gets transferred here.
So, there's a tensor with that element size, and the size of each element is--
I guess this-- I think this is float 32, so that would be 2--
sorry, for 4 bytes. And so that's the size in bytes.
Now this is a little bit subtle. So how many bytes are actually sent?
Or transferred? Sent, slash,, received? So each tensor sitting on a rank has size_bytes.
And it needs to send it to world_size minus 1 in other machines--
or ranks, rather. But there's a factor of 2.
So why is there a factor of 2? Because you're doing an all_reduce, remember? so you
need to send all the distinct elements into basically one
place, it needs to get summed up, and then that needs to go back to everyone.
So a rank needs to send the input out and then receive
the output. So that's why there's a factor of 2 there. And so the total duration is the world_size times
the actual duration that passed. So I guess we're just assuming that--
if there's four processors, that's like four times as much wall clock time that happened.
And the bandwidth is just the bytes over the duration. OK, so what do we get here?
Is about 277 gigabytes per second. So I think for at H100 above, I think
I claimed that there was something like 900 gigabytes per second. Now of course, as we know, your mileage
varies depending on the size of the tensors and the exact number of devices and the weather
and whatever-- no, not the weather, but various factors. So your mileage might vary.
So it's always good to benchmark to see what is actually the number of gigabytes
per second you're getting. OK. So reduce_scatter is going to be very, very similar,
so let's just go through this really quickly. So we created input, which is world_size times num_elements.
So each rank is going to have the matrix.
And so we're going to warm up, and then start the clock,
reduce_scatter, stop the clock, and then see how long it took.
Well, OK, that's not helpful. And then let's look at the bandwidth.
So the number of sent_bytes is a factor of 2 here
because in reduce_scatter, remember, all you're doing is you're sending your inputs into one place.
If you just think about reduce, all the elements just go into one place and that's it.
And scatter just means that different components of your tensor are going to different places,
but effectively it's like a reduce. So if you do the same calculation,
you'll see that it's-- I guess I get 70 in this case.
So I don't exactly know why it's exactly 70 as opposed to some other number.
I guess one could speculate that all_reduce generally there's more traffic that happens, and all_reduce is
our potentially more optimized. I think that NVIDIA hardware has this sharp acceleration that
actually does some of these computations in the actual network, which shaves off a factor of 2,
but I don't know if that's completely accounts for a difference here. There's a lot of stuff that happens in NCCL that it's
a little bit hard to reason about the performance exactly, hence benchmarking.
Yeah? I had a question about the satellites-- or the data bytes and how that was calculated specifically.
It looks like it calculates just the data that's being sent to the output, but what about the input
to the reduction step? I'm wondering like how it gets the inputs to the-- So the question is it seems like this is just
the bytes for the output and what about the input? So to be clear, I am assuming that the inputs just
are already on the device, so I'm not counting that time. And I'm just counting what needs to happen
to do the reduce_scatter. Is this just the scatter or--
This is a reduce_scatter operation. So you need [INAUDIBLE]?
So this function does reduce_scatter. So it's one operation.
I mean, like we [INAUDIBLE] twice in the previous
[INAUDIBLE] because we were doing reduction, which accounted for half of 2 bytes and then
[INAUDIBLE]. So you're saying that for all_reduce,
there's a 2x because you needed to reduce NV and you needed to spread out again?
For reduce_scatter, I mean, it's just a name. It's called reduce_scatter, but it's really
just a row reduction.
And you can also see based on this that if you do reduce_scatter and you do all_gather,
each of those doesn't have the factor of 2, so when you add them up, you get a factor of 2, which is another way to see that all_reduce is twice.
And there's some references you can go read about how
to benchmark and these collective operations. OK, so let's now talk about the distributed training piece.
So our general approach here is going to be-- I'm going to walk through a bare bones implementation of each strategy on deep MLPs, essentially.
So recall that you generally are in a regime where MLPs are the compute bottleneck and transformers, not
the attention. So in some ways, even though this is a very simple architecture, it's fairly representative of the type of workloads
that you'll see. So let's start with data parallelism.
Actually, just one note, is that data, tensor, and pipeline parallelism are-- you can just think about them as different ways of cutting up
either your model or your data, which hopefully I'll depict visually here.
OK, so in data_parallelism, here's your model, assuming it has four layers.
Each layer of the MLP is just a matrix multiply where this is the hidden dimension.
And so the data is also a matrix, which is-- there's the batch dimension and then the hidden dimension.
And data parallel just cuts along the batch dimension into essentially smaller pieces.
So now each rank is going to get a different slice of the data.
So let's do an example here. So I'm going to generate some sample data.
So let's say I have a batch size of 128, hidden dimension of 1024, and then
just generate some random data. So I have batch size by number of dimension,
and I'm going to run this data parallel algorithm, or DDP.
So here, I'm going to--
so I got passed this data. There's a batch size and the dimension as claimed
from before. Now I divide the batch_size by the world_size, so I get the local batch_size.
That's how many-- how big the batch size is on a given rank.
And then I'm going to, based on the rank, just figure out which starting and ending indices of size
local_batch_size I need to access, and then get the corresponding data from that.
So basically, I'm just reaching in and grabbing some subset of the rows based on the rank.
OK, so now I'm setting up the MLP here. And this is done very bare bones, you could say.
So here, I am creating the MLP parameters. So each layer has essentially a matrix,
which is num_dim by a num_dim, and remember, num_dim is 1024.
And I'm going to create the optimizer. So remember, this function is running asynchronously
on all the different-- on each rank. So each of the four ranks is going to be running this
with rank equals 0, 1, 2, 3. And now I'm going to start training.
So for a number of steps, I'm going to do a forward pass through the layers,
matrix multiply non-linearity, matrix multiply non-linearity. There's four layers here.
We're going to compute some loss. I don't really care what the loss is, it's just something made up. And I'm going to do the backward pass.
So so far, this just looks like I'm implementing SGD.
And that's the point. The only difference is now, to implement DDP, is that you just inject this line here which synchronizes
the gradients across workers. So what you do is for each of the layers, you call an all_reduce where you're averaging,
and the thing you're averaging is param-grad.
So it's just like you've hijacked someone's SGD code and you're saying, "Wait, I'm actually going to just mix all
the gradients after the backward pass." And then after you do that, you just
update the parameters as usual. So from the SGD perspective, it seems like nothing is happening,
I'm just running SGD, but someone has just mixed my gradients.
So I guess just to print out some things.
So data parallel, I'm printing out the loss. So one thing to note is that the losses are different between all
the different ranks because they have different datas, but after the all_reduce, all the parameters are the same.
So this is [INAUDIBLE] your textbook application
of all_reduce in ML setup. Yeah? [INAUDIBLE] each rank routes this [INAUDIBLE],
how do they ensure that they are all at the same step? Marching through the same step?
Or [INAUDIBLE] it doesn't matter [INAUDIBLE]? So the question is, how do you ensure-- if all of these processes are just running asynchronously,
how do you make sure that each of them is actually, for example, on the same step?
This is because all_reduce is a synchronization point. It'll stop everyone and do the all_reduce.
So you have to be careful because if one of your ranks
has a missing all_reduce, then it will just hang. [INAUDIBLE] others will be waiting for you--
Yeah. Yeah. Yeah? Why does getting the initial parameters depend on the rank?
The question is, why does getting initial parameters depend on the ran? Aren't they the same--
they should be-- They're the same. The reason is just because-- I guess I don't-- the code for this basically puts it on the appropriate GPU.
[INTERPOSING VOICES] OK. Any other questions?
So DDP is something you implement in assignment 2, which maybe some of you have looked at or maybe not.
It will be done in the context of a transformer, but this is the most bare bones version
so you can see very clearly what's happening.
OK. So that's DDP.
Losses are different across ranks, but the gradients are reduced to be all the same,
so therefore, the parameters of all the ranks are the same.
So actually, you're doing world_size number of SGD runs,
but because they're synchronized, they're doing the same thing. So you can think about this as an instantiation of analog
of activation checkpointing where sometimes you just do extra compute because you don't want to store things.
In this case, we could have, for example, ship the optimizer state around, but that would be a bad idea because it's much faster just to run the--
update the optimizer state than to actually move the optimizer
parameters around. OK. So last year, I did try to do FSDP,
but that was a hairball, so I'm going to skip that and do
a tensor parallel. So here, the picture is--
we leave the data the same. And now what we're going to do is we're going to cut the model along the hidden dimension.
So each rank is going to get every layer,
but it's going to get only part of each layer. And what we're going to end up doing
is transfer all the data and the activations around. OK.
So we're generating the same sample data, and let's look at tensor parallel.
OK, so I have the batch_size and num_dim as before.
And now, I'm going to-- before I was cutting batch_size, but now I'm cutting num_dim.
So I have local num_dim equals 1024 divided by world_size.
And that's 256. So each model essentially--
sorry. Each rank gets a part of the model, which is 1 over the world_size fraction of the parameters.
And remember, why we're doing parallelism at all is because the model won't be able to fit into a single GPU,
so we're going to shard it across multiple GPUs. So the parameter matrices are now num_dim by local_num_dim.
And now, each rank is going to-- I'm only going to implement the forward pass here,
not the whole training loop. So I'm going to start going through all the layers.
So I'm going to compute the activations first.
So this looks pretty normal, except for, remember, the activations are actually batch size by local_num_dim rather than num_dim
because I only each rank only has a fraction of the activations now.
But now, once I get the activations, I need to communicate.
And here, what I have to do is I'm
going to allocate memory for all the activations. So at this point, every one has an
x, but that x represents a different part
of the activations. So now I'm going to just allocate batch_size and local_num_dim, but world_size number.
So basically, each rank is going to basically have enough-- I'm going to just get--
basically have world_size number of batch_size
by local_num_dim matrices. And then I'm going to do an all_gather.
So I'm going to send all the activations.
And this-- I mean, it's fairly simple. So x, remember, is batch_size times local_num_dim,
but x is different for every rank. So when I do the all_gather, I'm going to put it in activations, which has
essentially a world_size number of the same shape as x.
So now every rank has the same activations--
now it has the activations of all the models-- of the whole model.
And then just to concatenate them together to get x.
OK, so now x is now, again, batch_size by num_dim
And I repeat. So as you can see, this is-- there's quite a bit
of communication that happens, which is why-- remember, Tatsu said that for tensor parallel,
you need pretty high interconnects, otherwise you'd be passing a lot of these activations around.
And then you do it for the next layer, and the next layer,
and you get the idea. And just to print out some output.
So tensor parallel, let's see here.
Forward pass produces activations of basically the full size, and everyone
has the same activations at the end.
So backward pass I'm going to skip because that's annoying to do.
All right. Any questions about that?
Yeah? I was just wondering why it's hard to do [INAUDIBLE]? So why is it hard to do the backward pass?
I don't think it's necessarily hard, but I guess in the constrained time and space--
it's not hard, it's just requires a bit more work.
OK. So now let's go to pipeline parallelism.
So in this case, we're cutting the model by layers.
So all the ranks get all the data and all the ranks--
each rank gets all of one layer, but they get different layers
OK, so sample the data, and run this function for all the ranks.
OK. So here, I'm going to figure out how many layers go
in each rank, which is 2 here.
So I have a four-layer network. I have two ranks, so each rank gets two of the layers--
just like this picture, actually. And here, I'm going to just allocate the parameters just
for the layers that I need.
So I'm going to do the forward pass. Remember, there was a further optimization that you do,
which is if you just do it naively,
you get these pipeline bubbles that Tatsu talked about before. One way to mitigate that is to break up
the batch into micro-batches. So here, I'm going to divide this batch
into batches of size 32. So 4 batches of size 32.
And then-- now the idea is that every rank is going to essentially wait for the previous rank
to pass it to the activations. It's going to apply those layers and then it's going to forward it to the next rank.
So starting at the base case, we have rank equals 0. That's just the data.
So I'm just chunking the data into a bunch of micro-batches.
And going through each of the micro-batches,
first I receive the tensor. So I'm using these point-to-point primitives
now instead of the collective primitives.
And I essentially-- they basically received the tensor x.
And then I'm going to compute the layers that are assigned to this rank. So in this case, there's only two of them.
And then I'm going to send it to the next rank.
And then, again, send is a point-to-point operation.
And then the next batch, I'm going to do the same thing,
so I'm going to skip that. OK, so that's basically it.
So pipeline parallel, at least the very naive version of it, is relatively conceptually simple.
But it's not to mention last time, there's many things that are missing from this basic implementation.
Overlapping the communication and computation is something we're not doing at all here.
For example, receive and send are synchronous, but you should really make them async. And also, the order in which you do the forward--
actually, this is just a forward, even, not the backward, but once you have the backward, then you have to figure out how to interleave the forward
and backward steps.
Yeah? [INAUDIBLE]-- I guess, maybe what you just mentioned about the async being shown here, is some--
I guess in actuality, like, the GPU would be listening whether another one passes something to it,
and it's [INAUDIBLE] it only starts
processing once the [INAUDIBLE] layer before it passes [INAUDIBLE]
So the question is, is this like event-driven programming where you're just waiting for things to happen?
And I think in event-driven programming, you basically write these handlers, and then whenever stuff happens, maybe you get a mouse click,
maybe you get a file ready event, then a piece of code runs. That's quite different, I think, from this style
of coding where everything has to work in lockstep.
It is true that you're waiting for the previous rank
to send you the information, but at least in this implementation, there's no flexibility of where it's getting from.
It's not like it's waiting for arbitrary data to come from anywhere. I think there are ways to do asynchronous training,
which was, I think, quite popular, more than 10 years ago,
where it's more event-driven where you have a server that
sends data, and whenever the gradients are ready, it just uploads and then the gradients get accumulated,
and if workers die, then that's-- then that's handled more robustly.
But in modern training, despite scaling up quite a bit,
everything seems to be in a synchronous paradigm.
Yeah, so it is true that when I say the workers are--
and the ranks are operating asynchronously, that's just because it's different processes. But you're still putting quite rigid synchronization
on how everything is working in lockstep. Yeah?
I'm just curious, how would you change this program to handle [INAUDIBLE] layers [INAUDIBLE]
computation with [INAUDIBLE] data? So the question is, how would you change this to overlap communication and computation?
So, for example, when you send this, there's no reason to just wait for the data to be sent.
You just basically fire off the send. Remember that the send actually happens
on the GPU via some kernel launch, so that's independent. And it can just go and process another micro-batch right away.
So the way, I think, you would do this is there's another function called i.send
which is asynchronous. Actually, this should be synchronous.
Asynchronous, which returns a handle. And so you basically do all the send, and then at the end,
you basically wait for all the sends to complete.
And then for overlapping the-- when you actually have the backward step,
then you basically have to schedule that in here.
Yeah? [INAUDIBLE]
So the question is, if you have multiple sends and multiple receives, how do you know which is which?
So here, the tensor name doesn't matter. It's just whatever variable is there.
And what you're specifying is the source. So if I'm at a node and I'm receiving, then whatever
the next message coming from that rank, I'm just going to put in this x and continue executing.
What if I want to do two sends from the same rank? If you want to do two sends from the same rank.
To the same destination? [INAUDIBLE]
So I'm not quite sure about this, but I think if you have two sends, it's put in a stream.
So the order of the sends still is preserved. It's just that other stuff can happen at the same time.
You can send to-- I think if you have a pair and do
two sends, then that order is preserved, but the order in which you send--
some other rank is sending to another rank, it can happen at any time.
Yeah? What would happen if you just did [INAUDIBLE], but then no one's receiving it [INAUDIBLE] gets up there--
So what happens if you send and no one receives it? I think it would just stop, we just wait.
Because there's no-- yeah. I mean, because--
I mean, the process could just be running and you don't know whether it will-- it's just--
I mean, it's just code executing. So you don't know if it's never going to get there or if it's just going to be a matter of time.
Yeah? [INAUDIBLE] So the question is, what happens to the last rank?
So at the end, the last rank has all the activation. So that has basically the results of a full forward pass.
And then if you implement the backward pass, then you would be actually now computing the gradient
with respect to the loss, and then you would go back down and send from rank to rank minus 1 and so on.
OK. I guess maybe-- I was afraid I was going to run out of time, but it looks like I actually have time.
Maybe next year I should do the backward pass. OK.
So actually, I'm going to finish quite early today, so if you have any other questions, you should ask.
So so far, we've gone through three simple examples of data, tensor, pipeline parallel.
Of course, this is for simple MLPs. You would actually want to do this with your own fancier model
like a transformer. I did argue that, at least at--
the core ideas you can understand through the MLP.
I think the-- but of course, when you want to train, you want to train transformer, not a deep MLP.
So you still have to implement the full complexity. What's also missing is the communication and computation
overlap, which is not really handled very carefully here. And there is generally a more complex code with bookkeeping.
I encourage you to check out Megatron-LM or PyTorch's FSDP.
It gets fairly hairy. And one of the things that, I think,
makes some of the bookkeeping, at least for, let's say, FSDP-- and you'll be exposed to this A2 a bit,
is that if you want something that handles arbitrary architectures, then you have to figure out the parameters
and do a bunch of bookkeeping to-- and figure out whether the layers are and so on.
Whereas in the MLP case, it's just I've made the decision that I'm going to split the model in this particularly simple way.
One other thing I'll just mention as an aside is that all of what we're doing in this course is PyTorch,
but it is useful to be aware of this whole other ecosystem around Jax and TPUs, which is actually nice in some way.
And the idea here is Jax allows you to just define the model.
It defines the sharding strategy, and then the Jax compiler handles the rest.
So there's this toolkit that we developed called Levanter based on Jax.
And I'll just show you a snippet of what happened. So this is FSDP and 10 lines of code.
And basically, have a model, and then you just say shard
with this particular-- I mean, I don't expect you to read this exactly, but basically, you define which dimension
you're going to shard by, and then that's it. And similarly for tensor parallel,
you're just saying I'm going to shard the model along the--
you can shard on the head dimension for attention,
and also, you can shard based on the model dimension. So in some sense, this gives you a conceptual simplicity
of what you're trying to do is you have this-- basically a computation graph, but it has these dimensions,
the model dimensions, the embedding dimension, the attention sequence dimension. And Jax allows you to basically just specify
which dimensions you want to cut by and also define a mapping from that onto the actual TPUs.
And then, the Jax compiler magically just figures out how to compile that
down into the primitives that shuffle things around. So this is much more higher level
than operating with a collective communication.
But we're sticking with PyTorch because it
allows you to see a underneath the hood what's actually happening. But if you're actually doing this in the real world,
obviously you don't need to-- and you probably shouldn't implement all of this from scratch. OK.
So that's the end of the Jax digression. So to just summarize, we've seen many ways to parallelize so far.
And each of these ways of parallelizing you can think about just like splitting either the model or the data along
some dimension-- either the data, the batch dimension, or the width dimension, or the depth dimension, or the context
length dimension. We also see this recurring theme of computation.
You can recompute something from scratch,
or you can store it in memory and suffer the data transfer cost.
Or, in now in a multi-GPU, multi-node setting, you can actually store on another GPU's memory
and then communicate, which is even slower.
So there's these trade-offs here. And often, re-computation actually can be better,
but obviously you can't compute the whole thing. And often, you're either communication- or
memory-limited. The final word is that--
it is the case that hardware is getting better. So you might think that, well, maybe none of this
is really necessary because in five years, everything will fit in L1 or HBM.
So this is not going to be the case because those might grow
quite a bit, although there are still physical limits, will always be ending up with bigger models that
are at the limit of what the hardware can do. So this hierarchical structure, ever since computer systems
was a thing, has always been with us, and it will always be there.
OK. That's all I have for you today. So I can take any questions.
Yeah? So [INAUDIBLE] the same set of parameters, the forward pass
may be different because you're not [INAUDIBLE] might be a function of the input data set?
So for example, when you have [INAUDIBLE], so then [INAUDIBLE]
training process [INAUDIBLE] So the question is in data parallel,
you're saying that even though the parameters are all synchronized, there could be other things that
depend on the data, like in batch norm? So I don't actually know how you--
batch norm is always kind of annoying. So I don't know exactly how you would do that
off the top of my head. And I guess, at least in the LM world,
that doesn't really show up because layer norm is used.
And as long as you initialize all them parameters and using the same random seed, you'll be fine.
I mean, there could be non-determinism issues on the GPU, but hopefully those are minor.
Yeah? [INAUDIBLE]
So the question is, does PyTorch have some niceties as well,
kind of like what Jax offers? Is that-- Yeah. So, I mean, PyTorch does have the FSDP Library,
which you should absolutely use if you're not taking this class. Which basically is a wrapper.
You define any model and it just does FSDP on it. I think that-- now, if you're asking how well it can more
custom-- allow you to more do custom sharding, I think there are some things that are coming,
but it's not as, I think, as developed. I mean, I think there's this, I think, spectrum between the Jax
world where you declaratively define things, and I think the Google infrastructure, if you stay within the Jax TPU system,
it's pretty well-developed. But then if you look at of DeepSeek,
which is the opposite end where you have these GPUs, which
actually really bad interconnect, which means that they have to go in and--
they actually go to the NCCL level and actually do a bunch of things, which I don't quite understand, to eke out the performance.
Whereas if you're running a Jax, you just, from on high, declare your model and then stuff happens.
So the ways that you leverage hardware I think really
depends on what ecosystem you're operating in. Yeah?
[INAUDIBLE]
Yeah. So the question is activation checkpointing what-- there is an API that basically allows you to--
I mean, I guess in PyTorch and Jax, too, specify which parts you want to recompute,
because clearly you don't want to recompute everything or nothing.
Probably every few layers, probably right after big MAMLs
where-- For example, if you have, let's say, a MAML and then pointwise nonlinearity,
I don't think you need to store two copies of-- but basically, if you have two things where
it's trivial to get to, then you might as well just
store one version. Yeah, over there.
[INAUDIBLE]
So the question is, are GPUs going to ever be replaced by a transformer-specific hardware?
So you're seeing this in the inference space quite a bit already with--
like, Grok and Cerebras have specialized hardware that can do inference-- and also, I guess, training,
Cerebras has training. So basically, those hardwares essentially
give you just a lot more on-chip memory. I mean, that's basically the name of the game.
I think Cerebras has a huge-- essentially effectively an L1 cache
so you don't have to move things off. And I think a lot of simplifications
can happen-- because GPUs were-- there's a lot of baggage, actually, if you think about-- because they were designed
in an era where you had to do a lot of branching and various types of ad hoc computations
which are not really needed in the deep learning regime. So I think there are quite a few opportunities
to improve the hardware as well. I think there was a hand back there, and I'll--
I don't know if this is like the right question but I'm thinking about, but in the context of the lecture,
it's basically a model that's being trained in one go that's being optimized, but I'm wondering
if any of the techniques that we're talking about can be used to incrementally train a model--
for example, as you get new training data, not just to fine-tune, but actually to recalculate
everything without having to recalculate everything. Yeah. So the question is, can these techniques
be used to essentially do continued training? Yeah, absolutely.
So if you think about the unit of what we're working on, is just doing gradient steps.
So if you take a half-trained checkpoint, you can just continue doing what this is.
There's nothing specific about starting from scratch here.
I think there's a question there. Yeah, so on the model-specific hardware,
[INAUDIBLE] previous question, presumably there's a physical technical reason
why you can't make nodes much larger than they are currently.
What's the change that you're talking about?
So if you could just make GPU nodes infinitely-- like as big as you want, people would do that.
So presumably there's a tech-- like a hardware reason that's not possible, so what's the actual advancement being
done for the [INAUDIBLE] specific hardware you mentioned? Yeah, so the question is there are physical limits for sure
for a GPU. Let me just go-- so you can make GPUs obviously infinitely large, or infinitely
dense. I mean, there's also power issues. You need to get rid of all the heat.
And, there's only so much bandwidth that can fit.
So I don't know the exact details, but at least in some of the Cerebras case--
I mean they have this way of manufacturing basically
the chips so that the memory is on the chip.
So I guess it's just a way of putting it on there. And I think that there are obviously
trade-offs because it comes at a cost of not having as much flexibility.
But in general, I think the way to maybe think about this more broadly is that GPUs were still
developing the CPU era where it's much more control-focused.
I have code that I'm executing, that's the first-class citizen, and then data needs to be moved to execute to handle the code.
But the big difference with deep learning workloads is that it's all data flow.
The computation graph, if you look at these, is static. You know from the beginning exactly all the computations
that are going to be done until essentially the end of training. So using that knowledge, you should
be able to lay out your computation in a much smarter way than having to deal with the flexibility
uncertainty over ad hoc computation.
OK. Maybe a few more questions and I'll end there. Yeah? Is the computational graph usually stored on the CPU
or in the GPU? So the question is, where is the computation graph stored?
Well, the code is-- I mean, all this code is running on this CPU.
But when you call something like a PyTorch function
that needs to run on GPU, then it launches kernels under the hood, and the kernels are code that runs on the GPU.
Yeah, I'm not sure of that--
so I guess maybe another answer is that the computation graph is more of, I guess, a conceptual.
It's not like there is a graph literally that's being-- I mean, I guess there is, but it's
not like the graph gets put on the GPU, if that makes sense.
OK. So these communication primitives that [INAUDIBLE] can receive and send [INAUDIBLE] CPU
instructions [INAUDIBLE]
So the question is, the communication primitives, are they CPU or GPU? So these collective operations are,
in some sense, abstract specification of what types of operations need to happen, which can happen.
If you remember, this PyTorch distributed has different backends.
So it could happen on a GPU or it happen on a CPU.
[INAUDIBLE] is it-- like, is the CPU scheduling them
or is it kernels which are independent? Yeah.
Well, the CPU drives-- basically is the master still.
And then when you do a collective operation, it calls the NCCL library, which launches,
which is it's still CPU, and then it launches some kernels that move data around.
Yeah. OK. Maybe this is a good place to end. All right. I will see you next Monday--
or Tuesday, rather. Thanks.
