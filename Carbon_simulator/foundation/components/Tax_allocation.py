import json
import numpy as np

from Carbon_simulator.foundation.base.base_component import (
    BaseComponent,
    component_registry,
)


@component_registry.add
class CarbonTaxation(BaseComponent):


    name = "CarbonTaxation"
    required_entities = ["Carbon_idx", "Carbon_project", "Taxation"]
    agent_subclasses = ["BasicMobileAgent", "BasicPlanner"]

    """
    Required methods for implementing components
    --------------------------------------------
    """

    def __init__(
            self,
            *base_component_args,
            planner_mode="inactive",
            total_idx=200,
            max_year_percent=100,
            base_carbon_price = 15,
            years_predefined = None,
            agents_predefined = None,
            total_fields=1600,
            taxrate=0,
            permits=0,
            percentage_of_carbon_projects=0.1,
            **base_component_kwargs
    ):
        super().__init__(*base_component_args, **base_component_kwargs)

        self.planner_mode = planner_mode
        assert self.planner_mode in ["inactive", "active"]

        self.total_idx = float(total_idx)
        assert self.total_idx >= 1

        self.total_fields = int(total_fields)
        assert self.total_fields >= 0

        self.step_count = 0
        self.percentage_of_carbon_projects = float(percentage_of_carbon_projects)
        assert 0 <= self.percentage_of_carbon_projects <= 1

        self.base_carbon_price = float(base_carbon_price)
        assert self.base_carbon_price >= 0

        self.max_year_percent = int(max_year_percent)
        assert 0 <= self.max_year_percent <= 100

        self.years_predefined = str(years_predefined)
        self.agents_predefined = str(agents_predefined)

        self.log = []


    def get_n_actions(self, agent_cls_name):
        """This component is passive: it does not add any actions."""
        if agent_cls_name == "BasicPlanner":
            if self.planner_mode == "active":
                ### Action space ###
                # num one: taxrate
                # num two: carbon project placement
                # select percentage from 0 to 100
                return [
                    ("Carbon_{:03d}".format(int(r)), 101)
                    for r in range(2)
                ]
        return 0

    def get_additional_state_fields(self, agent_cls_name):
        """This component does not add any state fields."""
        if agent_cls_name == "BasicMobileAgent":
            return {}
        if agent_cls_name == "BasicPlanner":
            return {
                    "year_num": 0,
                    "env_idx": [0 * self._episode_length / self.period],
                    "tax_rate": [0 * self._episode_length / self.period],
                    "average_Er": 1
                    }
        raise NotImplementedError

    def component_step(self):

        world = self.world
        world.planner.state["year_num"] = self.world.timestep // self.period

        # punishment at end of years#
        if self.world.timestep % self.period == 0:
            #add tax directly after incurrence
            """for agent in world.agents:
                if agent.state["inventory"]["Carbon_idx"] > 0:
                    taxation =  agent.state["inventory"]["Carbon_idx"] *self.base_carbon_price * (world.planner.state["tax_rate"]/100)
                    agent.state["inventory"]["Coin"] -= taxation"""

            sum_Er = 0
            for agent in world.agents:
                sum_Er += agent.state["Carbon_emission_rate"]

            world.planner.state["average_Er"] = sum_Er / world.n_agents
            assert 0 <= world.planner.state["average_Er"] <= 1

            self.log.append({
                "emissions_per_agent": self.world.planner.state["emissions_per_agent"],
            })


        # divided idx at start of years#
        elif world.timestep % self.period == 1:
            for agent in world.agents:
                if agent.state["inventory"]["Carbon_idx"] > 0:
                    self.world.planner.state["emissions_per_agent"][agent.idx] += agent.state["inventory"]["Carbon_idx"]
                    # when in the negative, the overspending of emissions gets logged per agent

            if self.planner_mode == "active":
                idx_action = []
                total_percent = 0
                for i in range(2):
                    # 0 - 100
                    planner_action = self.world.planner.get_component_action(
                        self.name, "Carbon_{:03d}".format(int(i))
                    )
                    if i == 0:
                        # allocation of percentage of green development
                        world.planner.state["env_idx"] = self.total_fields * self.percentage_of_carbon_projects * (planner_action/100)

                    elif i == 1:
                        world.planner.state["tax_rate"] = planner_action

                '''if sum(idx_action) > world.planner.state["remained_idx"]:
                    idx_action = [ia * int(world.planner.state["remained_idx"]) // sum(idx_action) for ia in idx_action]'''
                world.planner.state["remained_idx"] -=  sum(self.world.planner.state["emissions_per_agent"])

                for agent in world.agents:
                    agent.state["inventory"]["Carbon_idx"] = 0
                    agent.state["escrow"]["Carbon_idx"] = 0
                    agent.state["endogenous"]["Taxation"]= self.base_carbon_price * (world.planner.state["tax_rate"]/100)

            elif self.planner_mode == "inactive":
                world.planner.state["env_idx"] = self.total_fields * self.percentage_of_carbon_projects * (
                            self.permits / 100)

                world.planner.state["tax_rate"] = self.taxrate
                world.planner.state["remained_idx"] -= sum(self.world.planner.state["emissions_per_agent"])

                for agent in world.agents:
                    agent.state["inventory"]["Carbon_idx"] = 0
                    agent.state["escrow"]["Carbon_idx"] = 0
                    agent.state["endogenous"]["Taxation"] = self.base_carbon_price * (
                                self.taxrate / 100)

            else:
                assert self.planner_mode in ["inactive", "active"]

            # Decided Punishment
            assert world.planner.state["tax_rate"] >= 0

            # Divide Carbon_idx to env
            assert world.planner.state["env_idx"] >= 0

            # layout the Carbon_project
            '''world.maps.set("Carbon_project", np.zeros(world.world_size))
            world.maps.set("Carbon_projectSourceBlock", np.zeros(world.world_size))'''
            empty = world.maps.empty
            project_num = 0
            n_tries = 0
            world.planner.state["remained_permits"] -= world.planner.state["env_idx"]

            while project_num < world.planner.state["env_idx"]:
                n_tries += 1
                if n_tries > 200:
                    raise TimeoutError
                r = np.random.randint(world.world_size[0])
                c = np.random.randint(world.world_size[1])
                if empty[r, c]:
                    world.maps.set_point("Carbon_project", r, c, 1)
                    world.maps.set_point("Carbon_projectSourceBlock", r, c, 1)
                    empty = world.maps.empty
                    project_num += 1

            self.log.append({
                "env_idx": world.planner.state["env_idx"],
                "emissions_per_agent": self.world.planner.state["emissions_per_agent"],
            })
            self.wandb_log()

        else:
            self.log.append([])

    def generate_observations(self):
        """This component does not add any observations."""
        obs_dict = dict()
        for agent in self.world.agents:
            obs_dict[agent.idx] = {
                "year_num": self.world.planner.state["year_num"],
                "average_Er": self.world.planner.state["average_Er"],
            }
        '''
        obs_dict[self.world.planner.idx] = {
            "punishment": self.world.planner.state["punishment"],
            "year_num": self.world.planner.state["year_num"],
            "env_idx": self.world.planner.state["env_idx"],
            "mobile_idx": self.world.planner.state["mobile_idx"],
            "agents_volume": [agent.state["Manufacture_volume"] for agent in self.world.agents],
            "settlement_idx": self.world.planner.state["settlement_idx"],
        }
        '''

        obs_dict[self.world.planner.idx] = {
            "year_num": self.world.planner.state["year_num"],
            "agents_Research_ability": [agent.state["Research_ability"] for agent in self.world.agents],
            "agents_Coin": [a.state["inventory"]["Coin"] for a in self.world.agents],
            "agents_volume": [agent.state["Manufacture_volume"] for agent in self.world.agents],
            "agents_labour": [agent.state["endogenous"]["Labor"] for agent in self.world.agents],
            "agents_carbon_idx": [a.state["inventory"]["Carbon_idx"] for a in self.world.agents],
            "agents_emission_rate": [agent.state["Carbon_emission_rate"] for agent in self.world.agents],
            "emissions_per_agent": self.world.planner.state["emissions_per_agent"],
            "average_Er": self.world.planner.state["average_Er"],
            "tax_rate": self.world.planner.state["tax_rate"],
            "env_idx": self.world.planner.state["env_idx"],
            "carbonproject_percentage": self.world.planner.state["env_idx"],
            "avg_emission_rate": self.world.planner.state["average_Er"],
            "emissions_per_agent_mean": np.mean(self.world.planner.state["emissions_per_agent"]),

        }

        return obs_dict

    def generate_masks(self, completions=0):
        """Passive component. Masks are empty."""
        masks = {}
        if self.planner_mode == "inactive":
            masks = {}
        elif self.planner_mode == "active":
            masks = super().generate_masks(completions=completions)
            for k, v in masks[self.world.planner.idx].items():
                if self.world.timestep % self.period == 0:
                    remained_idx_mask = np.ones_like(v)
                    remained_idx_mask[0] = 0
                else:
                    remained_idx_mask = np.zeros_like(v)
                    remained_idx_mask[0] = 1
                '''remained_idx_mask[int(self.world.planner.state["remained_idx"]):] = 0'''
                masks[self.world.planner.idx][k] = remained_idx_mask
        else:
            assert self.planner_mode in ["inactive", "active"]
        return masks

    def additional_reset_steps(self):

        world = self.world
        #world.planner.state["punishment"] = self.fixed_punishment if self.fixed_punishment else 100  # 10, 30
        world.planner.state["year_num"] = 0

        world.planner.state["remained_idx"] = float(self.total_idx)
        world.planner.state["remained_permits"]= self.percentage_of_carbon_projects * self.total_fields

        world.planner.state["emissions_per_agent"] = np.zeros(self.n_agents)

        self.log = []
        world.planner.state["tax_rate"] = 0

        world.planner.state["average_Er"] = 1

        world.planner.state["env_idx"] = 0

        #world.planner.state["mobile_idx"] = [0] * self.n_agents

    def get_dense_log(self):
        if self.planner_mode == "inactive":
            return None
        elif self.planner_mode == "active":
            return self.log
    def wandb_log(self):
        world = self.world
        result = {
            "carbonproject_percentage": world.planner.state["env_idx"],
            "avg_emission_rate": world.planner.state["average_Er"],
            "tax_rate": world.planner.state["tax_rate"],
            "emissions_per_agent_mean": np.mean(world.planner.state["emissions_per_agent"]),
            "remained_emissons": world.planner.state["remained_idx"],
            "remained_permits": world.planner.state["remained_permits"]
        }

        for a in world.agents:
            result.update(
                {
                    f"agent_{a.idx}/emission_rate": a.state["Carbon_emission_rate"],
                    f"agent_{a.idx}/volume": a.state["Manufacture_volume"],
                    f"agent_{a.idx}/emission": a.state["Last_emission"],
                    f"agent_{a.idx}/taxation": a.state["endogenous"]["Taxation"],
                    f"agent_{a.idx}/inventory": a.state["inventory"]["Carbon_idx"],
                    f"agent_{a.idx}/escrow": a.state["escrow"]["Carbon_idx"],
                    f"agent_{a.idx}/labor": a.state["endogenous"]["Labor"],
                    f"agent_{a.idx}/coin": a.state["inventory"]["Coin"],
                    f"agent_{a.idx}/research_ability": a.state["Research_ability"],
                }
            )

        # Write the result to the log file
        with open("./taxation.json", "w") as log_file:
            json.dump(result, log_file)

        return result