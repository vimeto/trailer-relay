"""No-API direct-control example; deliberately simple, not a trained policy."""
import json
from relay.observe import observe
from relay.perception import attach
from relay.seeder import load_or_generate_relay
from relay.sim import Leg, LegConfig


def main():
    relay = load_or_generate_relay("mini", 0, "tracks")
    leg = Leg(relay["tracks"][0], config=LegConfig(time_budget=15))
    attach(leg, seed=0, level="medium")
    steps = 0
    while not leg.done:
        obs = observe(leg, "driver")  # the policy receives this, not leg.state
        leg.set_controls(0.0, -0.5)  # a blind straight reverse, for integration only
        leg.control_step(0.1)
        steps += 1
    print(json.dumps({"steps": steps, "observation_keys": sorted(obs), "metrics": leg.metrics()}, indent=2))


if __name__ == "__main__":
    main()
