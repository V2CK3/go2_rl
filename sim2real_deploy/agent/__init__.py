"""Agent：StateBuilder + LCMAgent + 部署配置。"""

from sim2real_deploy.agent.deploy_cfg import Go2BaseDeployCfg

__all__ = ["Go2BaseDeployCfg", "LCMAgent", "StateBuilder"]


def __getattr__(name):
    if name == "LCMAgent":
        from sim2real_deploy.agent.lcm_agent import LCMAgent
        return LCMAgent
    if name == "StateBuilder":
        from sim2real_deploy.agent.state_builder import StateBuilder
        return StateBuilder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
