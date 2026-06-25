# Planner-Executor-Reviewer with LangGraph

This folder contains a simple but real multi-agent orchestration using LangGraph.

## What this workflow does

1. `planner` breaks the task into steps.
2. `executor` works on the current step.
3. `reviewer` checks the result.
4. If the reviewer rejects it, the graph loops back to the executor with feedback.
5. If the reviewer approves it, the graph advances to the next step.
6. When all steps are approved, the graph finalizes the result.

## Why LangGraph fits this well

LangGraph is useful here because the orchestration logic is explicit:

- Nodes represent work.
- Edges represent allowed transitions.
- Conditional edges decide whether to retry, advance, or fail.
- State carries memory across the whole workflow.

That is much easier to reason about than hiding everything in one big prompt.

## State fields worth studying

- `plan`: The list of steps from the planner.
- `current_step_index`: Which step is active now.
- `current_output`: The executor's latest attempt.
- `current_feedback`: The reviewer's feedback for the next retry.
- `revision_count`: How many retries happened on the current step.
- `execution_history`: Every executor attempt.
- `review_history`: Every reviewer decision.
- `completed_steps`: Approved step outputs.
- `final_result`: The stitched output at the end.

## Persistence in this example

The workflow is compiled with `InMemorySaver`, which gives thread-level memory in the current Python process.

That means:

- Reusing the same `thread_id` preserves state while the process is alive.
- Restarting Python clears that in-memory state.

For production-style durability, the next step would be a persistent checkpointer such as SQLite or Postgres.

## Run it

```python
from agents.orchestrator.langgraph_workflow import run_task, get_thread_state

result = run_task(
    task="Write a short blog on microservices",
    thread_id="blog-1",
    max_revisions=2,
)

print(result["final_result"])
print(get_thread_state("blog-1"))
```

## Best lesson from this pattern

Keep each agent focused and dumb in a good way.
Put coordination, retries, memory, and stopping rules in the graph layer.
