"""The LangGraph agent.

    schemas    routing/grading decision models and AgentState
    prompts    every prompt, including the two calibrated ones
    decisions  router and grader, with recovery from structured-output failures
    nodes      node bodies, as closures over injectable clients
    graph      wiring and compilation
    diagram    render the compiled graph to mermaid or PNG
"""
