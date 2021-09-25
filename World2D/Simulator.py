class Simulator2D:
    def __init__(self, param=None):
        if param is not None:
            self.param = param
    def Simulate(self, param):
        Agents = param.Agents
        World = param.World
        for Agent in Agents:
            Agent.InitInWorld(World)
        for TimeIndex in range(param.SimulationStepNum):
            Actions = []
            for Agent in Agents:
                Observation = self.GenerateObservation(Agent, World)
                Agent.UpdateObervation(TimeIndex, Observation)
                Action = Agent.Action()
                Actions.append(Action)
            self.UpdateWorld(Agents, Actions, World)
    def GenerateObservation(self, Agent, World):
        return
    def UpdateWorld(self, Agents, Actions, World):
        return

__MainClaa__ = Simulator2D