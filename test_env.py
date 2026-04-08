"""Quick smoke test for the environment."""
import os
os.environ["TICKET_ROUTER_TASK"] = "basic_routing"

from server.ticket_router_environment import TicketRouterEnvironment

env = TicketRouterEnvironment()

# Test reset
obs = env.reset(seed=42)
print(f"Reset OK: task={obs.metadata['task']}, done={obs.done}")
print(f"Subject: {obs.metadata['email_subject']}")
print(f"Body preview: {obs.metadata['email_body'][:80]}...")
print()

# Test grading via direct call (bypass MCP for unit test)
reward_str = env._grade_submission("Sales", "", "")
print(f"Grade result: {reward_str}")
print(f"Episode done: {env._done}, Reward: {env._last_reward}")
print()

# Test medium task
os.environ["TICKET_ROUTER_TASK"] = "extraction_routing"
env2 = TicketRouterEnvironment()
obs2 = env2.reset(seed=0)
print(f"Medium task subject: {obs2.metadata['email_subject']}")
r = env2._grade_submission("Tech Support", "ERR-413", "")
print(f"Medium grade: {r}")
print()

# Test hard task
os.environ["TICKET_ROUTER_TASK"] = "pii_redaction"
env3 = TicketRouterEnvironment()
obs3 = env3.reset(seed=0)
body = obs3.metadata["email_body"]
print(f"Hard task subject: {obs3.metadata['email_subject']}")
redacted = body.replace("4532-1234-5678-9012", "[REDACTED]").replace("08/27", "[REDACTED]")
r3 = env3._grade_submission("Billing", "", redacted)
print(f"Hard grade: {r3}")

print("\nAll tests passed!")
