from server.ticket_router_environment import TicketRouterEnvironment, TicketAction

def test_easy():
    env = TicketRouterEnvironment()
    obs = env.reset(difficulty="easy")
    assert obs.t_id == "T-101", obs.t_id
    assert obs.tier == "Standard", obs.tier
    assert not obs.done
    
    act = TicketAction(act_type="route", dept="Sales")
    obs2 = env.step(action=act)
    assert obs2.done
    assert obs2.reward == 0.8, obs2.reward

def test_medium():
    env = TicketRouterEnvironment()
    obs = env.reset(difficulty="medium")
    assert obs.t_id == "T-201"
    
    act1 = TicketAction(act_type="search", query="ERR-77X")
    obs2 = env.step(action=act1)
    assert not obs2.done
    assert "postgres" in obs2.search_results
    
    act2 = TicketAction(act_type="route", dept="Database")
    obs3 = env.step(action=act2)
    assert obs3.done
    assert obs3.reward == 0.7, obs3.reward

def test_hard():
    env = TicketRouterEnvironment()
    env.reset(difficulty="hard")
    act = TicketAction(act_type="route", dept="Security")
    obs = env.step(action=act)
    assert obs.done
    assert obs.reward > 0.0

if __name__ == "__main__":
    test_easy()
    test_medium()
    test_hard()
    print("OK")
