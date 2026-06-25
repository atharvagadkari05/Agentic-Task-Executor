from agents.orchestrator.langgraph_workflow import run_task, get_thread_state

result = run_task(
    task="Write a short blog on microservices",
    thread_id="blog-1",
    max_revisions=2,
)

print(result["final_result"])
print(get_thread_state("blog-1"))