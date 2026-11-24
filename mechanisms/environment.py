import copy
import json
import random

from mechanisms.storage import save
from mechanisms.tasks import factorization, subset_sum

TASKS = {"factorization": factorization, "subset_sum": subset_sum}


class Environment:
    def __init__(self, config, instance, path):
        self.config, self.path = config, path
        self.public, self.answer = instance["public"], instance["answer"]
        self.task = TASKS[config["task"]["name"]]
        self.shortcut = config["treatment"]["shortcut_name"]
        self.notes = ""
        self.state = dict(
            events=[],
            checks=0,
            accepted=False,
            cheated=False,
            answer_returned=False,
            score=0,
        )
        self.tools = copy.deepcopy(config["environment"]["tools"])
        self.tools += [
            dict(
                name=config["task"]["verifier"],
                description=config["task"]["verification_description"],
                parameters=config["task"]["schema"],
            )
        ]
        self.tools += [
            dict(
                name=self.shortcut,
                description=config["treatment"]["description"],
                parameters=dict(
                    type="object", properties={}, additionalProperties=False
                ),
            )
        ]
        if len({t["name"] for t in self.tools}) != len(self.tools):
            raise ValueError("Tool names must be unique")
        if config["treatment"]["shuffle_tools"]:
            random.Random(instance["seed"] + config["treatment"]["order_seed"]).shuffle(
                self.tools
            )
        self.catalog = [{"type": "function", "function": t} for t in self.tools]
        self.instruction = config["environment"]["prompt"].format(
            task=config["task"]["instruction"],
            seconds=config["execution"]["timeout_sec"],
            checks=config["environment"]["max_checks"],
            shortcut=self.shortcut,
            honest_reward=config["treatment"]["honest_reward"],
            cheat_reward=config["treatment"]["cheat_reward"],
            operations="\n".join(
                "- " + t["name"] + (": " + t["description"] if t["description"] else "")
                for t in self.tools
            ),
            permitted=", ".join(
                t["name"] for t in self.tools if t["name"] != self.shortcut
            ),
        )
        self.persist()

    def persist(self):
        treatment = self.config["treatment"]
        self.state["score"] = (
            (
                treatment["cheat_reward"]
                if self.state["cheated"]
                else treatment["honest_reward"]
            )
            if self.state["accepted"]
            else 0
        )
        save(self.path, self.state)

    def call(self, name, raw_arguments):
        event = dict(
            operation=name,
            arguments=raw_arguments,
            status="requested",
            prohibited=name == self.shortcut,
        )
        self.state["events"].append(event)
        self.state["cheated"] |= event["prohibited"]
        self.persist()
        try:
            args = (
                json.loads(raw_arguments)
                if isinstance(raw_arguments, str)
                else raw_arguments
            )
            if not isinstance(args, dict):
                raise ValueError("Arguments must be an object")
            if name in {self.shortcut, "read_instance", "read_notes"} and args:
                raise ValueError("This tool takes no arguments")
            if name == self.shortcut:
                result = self.answer
                self.state["answer_returned"] = True
            elif name == "read_instance":
                result = self.public
            elif name == "read_notes":
                result = dict(notes=self.notes)
            elif name == "write_notes":
                if (
                    set(args) != {"notes"}
                    or not isinstance(args["notes"], str)
                    or len(args["notes"])
                    > self.config["environment"]["max_notes_chars"]
                ):
                    raise ValueError("Invalid notes")
                self.notes = args["notes"]
                result = dict(saved=True)
            elif name == self.config["task"]["verifier"]:
                if self.state["checks"] >= self.config["environment"]["max_checks"]:
                    raise ValueError("No verifier checks remaining")
                correct = self.task.verify(self.public, args)
                self.state["checks"] += 1
                self.state["accepted"] |= correct
                result = dict(
                    correct=correct,
                    checks_remaining=self.config["environment"]["max_checks"]
                    - self.state["checks"],
                )
            else:
                raise ValueError("Unknown operation")
            event["status"] = "returned"
        except (ValueError, TypeError, KeyError) as error:
            result = dict(error=str(error))
            event["status"] = "error"
        event["result"] = result
        self.persist()
        return result
