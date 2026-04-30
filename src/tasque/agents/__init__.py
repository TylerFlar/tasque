"""Cross-agent infrastructure shared by worker / coach / planner / strategist.

Houses the result inbox — a transient DB-backed channel used to ferry
an LLM's structured output back to the Python side via an MCP tool
call instead of post-hoc JSON parsing of the model's text output.
"""
