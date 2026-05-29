"""Abstract execution environment for agentic rollout.

An Env receives actions (type + args) and returns observations.
Subclasses implement domain-specific logic (code exec, web search, QA, etc.).

Reference implementation::

    class SandboxEnv(Env):
        def step(self, action_type, action_args):
            if action_type == "execute_code":
                return self._exec_code(action_args["code"])
            elif action_type == "finish":
                return {"observation": "", "done": True}
            ...
"""

from abc import ABC, abstractmethod


class Env(ABC):
    """Execution environment for one episode.

    Communicates via plain dicts for flexibility::

        obs = env.reset(task)
        while not obs["done"]:
            action_type, action_args = agent.act(obs)
            obs = env.step(action_type, action_args)
    """

    @abstractmethod
    def reset(self, task: dict) -> dict:
        """Start a new episode.

        Returns::

            {"observation": str, "done": False}
        """

    @abstractmethod
    def step(self, action_type: str, action_args: dict) -> dict:
        """Execute an action.

        Returns::

            {"observation": str, "done": bool, "info": dict | None}
        """

    def close(self):
        """Release resources (subprocess, connections, etc.). Override as needed."""
