/*=====================================================================
GaussianSplatPoolTask.h
-----------------------
Copyright Glare Technologies Limited 2026 -
=====================================================================*/
#pragma once


#include "../utils/Task.h"
#include "../utils/TaskManager.h"
#include "../utils/AtomicInt.h"
#include "../utils/Reference.h"


/*=====================================================================
GsPoolTask
----------
SESSION093: pool priority between the splat stages that share the main task manager, without touching the task manager.

The task manager is a plain FIFO: whoever enqueued first holds the workers until its tasks return. The render
traversal's expand (GsExpandTask in GaussianSplatRenderer.cpp) enqueues one task per thread, each pulling seeds off a
cursor until the cursor runs dry - so for the whole of an expand, every worker is taken. Anything enqueued meanwhile -
a saturation build kicked on its own cadence, that build's walk/gather/grid phases, a sort, a filter, the next
traversal - queues BEHIND them and waits for the cursor to run dry: measured session092 as par=1.00 w=1 (one worker out
of seventeen) for the barrier build's tree walk on every exterior capture, with ~8-10x of potential behind it.

The expand is the LONG job (~700-1600ms of work per full traversal) and everything else is short (a build is
~65-110ms of work, its phases a few ms each), so the expand yields: between seeds, a GsExpandTask checks
gsExpandShouldYield() and, if anything else is waiting for a worker, puts itself at the back of the queue - behind
whatever was waiting - and returns. FIFO does the rest. Shortest-job-first, with the hand-over latency bounded by one
expand seed (expand_seed_max_ms, which updateExpandSeedTarget() keeps near a thread's fair share).

What "anything else is waiting" means is this counter: our own tasks, dispatched and not yet picked up by a thread.
Every splat task that goes to the main task manager other than the expand derives from GsPoolTask and is dispatched
through gsAddPoolTask()/gsRunPoolTaskGroup(), which is what keeps the two ends of the count matched: +N at dispatch,
-1 as each task starts, on whichever thread runs it (runTaskGroup() runs some on the calling thread - those decrement
too, through the same run()). It counts QUEUED, not RUNNING: the signal clears the instant the last waiting task has a
thread, and from then on the expand takes back every worker those tasks release. That is the difference between the
price being the other stage's CPU time and its wall time - the first cut of session093 held a claim for a whole phase,
and the pool sat idle behind each phase's slowest task while the traversal ran serially (60-80ms per affected
traversal, interior).

Not counted, deliberately: tasks that are not ours to instrument - Sort::radixSortWithParallelPartition()'s own tasks
inside the traversal's and the sort task's sorts, and anything else the engine puts on the same pool. Those still wait
as they always did. (The barrier build's walk sort was the one such phase on the latency-critical path; it runs
serially now - see occluderTreeWalk().)

The count is process-wide - there is one main task manager, and two clouds' builds or traversals may overlap.
=====================================================================*/
inline glare::AtomicInt gs_splat_pool_queued(0);


class GsPoolTask : public glare::Task
{
public:
	virtual void run(size_t thread_index) override final { gs_splat_pool_queued.decrement(); runQueued(thread_index); }
	virtual void runQueued(size_t thread_index) = 0; // What run() used to be. Call THIS, never run(), to execute one of these inline without dispatching it.
};


// The only two ways a GsPoolTask should reach the task manager - see above.
inline void gsAddPoolTask(glare::TaskManager& task_manager, GsPoolTask* task)
{
	gs_splat_pool_queued.increment();
	task_manager.addTask(glare::TaskRef(task));
}

inline void gsRunPoolTaskGroup(glare::TaskManager& task_manager, const glare::TaskGroupRef& group)
{
	gs_splat_pool_queued += (int64)group->tasks.size();
	task_manager.runTaskGroup(group);
}

// > 0, not != 0: a miscount downwards (a GsPoolTask run through run() without a dispatch) must not leave the expand
// yielding forever.
inline bool gsExpandShouldYield() { return gs_splat_pool_queued.getVal() > 0; }
