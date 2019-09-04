# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        #distance of food from successor
        foodDist = []
        for foodPos in newFood.asList():
            foodDist.append(manhattanDistance(newPos, foodPos))
        #distance from nearest food
        minFoodDist = 0
        if len(foodDist) > 0:
            minFoodDist = min(foodDist)
        if len(currentGameState.getFood().asList()) > len(foodDist):
            minFoodDist = 0
            
        #distance of ghost from sucessor
        ghostDist = []
        for ghost in newGhostStates:
            ghostDist.append(manhattanDistance(newPos, ghost.getPosition()))        
        #if sucessor is 1 away from ghost, do not go
        minGhostDist = min(ghostDist)
        if minGhostDist <= 1:
            return float("-inf")       

        #if the action is to stop, punish sucessor
        if action == Directions.STOP:
            minFoodDist += 10
        
        return successorGameState.getScore() - minFoodDist

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        choice = None
        bestScore = float("-inf")
        
        #get minMax of every action
        for action in gameState.getLegalActions(0):
            succState = gameState.generateSuccessor(0,action)
            score = self.minMax(1, range(gameState.getNumAgents()), succState, self.depth)
            if score > bestScore:
                bestScore = score
                choice = action
        return choice

    def minMax(self, agent, agentList, gameState,depth):
        #if state is at the end
        if gameState.isWin() or gameState.isLose() or depth <= 0:
            return self.evaluationFunction(gameState)
        
        if agent == 0:
            s = float("-inf")
        else:
            s = float("inf")
        
        #get sucessor for each agents
        successors = []
        for action in gameState.getLegalActions(agent):
            successors.append(gameState.generateSuccessor(agent, action))
        #find max/min score for each sucessors of each agents
        for successor in successors:
            if agent == 0:                      #if first agent
                s = max(s, self.minMax(agentList[agent+1], agentList, successor, depth))
            elif agent == agentList[-1]:        #if last agent in this agent list
                s = min(s, self.minMax(agentList[0], agentList, successor, depth - 1))
            else:                               #goes to next agent
                s = min(s, self.minMax(agentList[agent+1], agentList, successor, depth))
        return s      

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        choice = None
        bestScore = float("-inf")
        alpha = float("-inf")
        beta = float("inf")
        #get alphaBeta of every action
        for action in gameState.getLegalActions(0):
            succState = gameState.generateSuccessor(0,action)
            score = self.alphaBeta(1, range(gameState.getNumAgents()), succState, self.depth, alpha, beta)
            alpha = max(alpha, score)
            if bestScore < score:
                bestScore = score
                choice = action
        return choice

    def alphaBeta(self, agent, agentList, gameState, depth, alpha, beta):
        #if state is at the end
        if gameState.isWin() or gameState.isLose() or depth <= 0:
            return self.evaluationFunction(gameState)
    
        if agent == 0:
            s = float("-inf")
        else:
            s = float("inf")
          
        #get sucessor for agents, end loop if value already less/more than alpha/beta
        actions = gameState.getLegalActions(agent)
        for action in actions:
            successor = gameState.generateSuccessor(agent, action)
        
            if agent == 0:                      #if first agent
                s = max(s, self.alphaBeta(agentList[agent+1], agentList, successor, depth, alpha, beta))
                alpha = max(alpha, s)
                if s > beta:                    #end loop if score more than beta
                    return s
        
            elif agent == agentList[-1]:        #if last agent in this agent list
                s = min(s, self.alphaBeta(agentList[0], agentList, successor, depth - 1, alpha, beta))
                beta = min(beta, s)
                if s < alpha:                   #end loop if score less than than alpha
                    return s
            
            else:                               #goes to next agent
                s = min(s, self.alphaBeta(agentList[agent+1], agentList, successor, depth, alpha, beta))
                beta = min(beta, s)
                if s < alpha:                   #end loop if score less than than alpha
                    return s

        return s
    
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        choice = None
        bestScore = float("-inf")
        for action in gameState.getLegalActions(0):
            succState = gameState.generateSuccessor(0,action)
            score = self.expectimaxMin(succState, 0, 1)
            if bestScore < score:
                bestScore = score
                choice = action
        return choice
    
    def expectimaxMin(self, gameState, depth, ghostAgent):
        #if state is at the end
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
            
        s = 0
        
        actions = gameState.getLegalActions(ghostAgent)
        for action in actions:
            if (ghostAgent == gameState.getNumAgents() - 1):
                s += self.expectimaxMax(gameState.generateSuccessor(ghostAgent, action), depth + 1)
            else:
                s += self.expectimaxMin(gameState.generateSuccessor(ghostAgent, action), depth, ghostAgent + 1)
        return s / float(len(gameState.getLegalActions(ghostAgent)))
                
    def expectimaxMax(self, gameState, depth):
        #if state is at the end
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        
        s = float("-inf")
        
        actions = gameState.getLegalActions(0)
        for action in actions:
            a = self.expectimaxMin(gameState.generateSuccessor(0, action), depth, 1)
            s = max(s, a)
        return s        
    
def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: 
      find position of pacman
      calculate distance of each food from pacman
      find distance from nearest food
      find all the position of ghost and their scared time
      if by the estimated time, the ghost can come to pacman without being scared, then consider it a valid ghost
      if the ghost is scared while near the pacman, ignore the ghost
      if pacman is 0 away from ghost, do not go this way
      return the current score minus the estimated time to get to the closest food
    """
    "*** YOUR CODE HERE ***"
    #position of pacman
    pacPositon = currentGameState.getPacmanPosition()
    #distance of food from successor
    foodDist = []
    for foodPos in currentGameState.getFood().asList():
        foodDist.append(manhattanDistance(pacPositon, foodPos))
    #distance from nearest food
    minFoodDist = 0
    if len(foodDist) > 0:
        minFoodDist = min(foodDist)
    if len(currentGameState.getFood().asList()) > len(foodDist):
        minFoodDist = 0
            
    #distance of ghost from sucessor
    ghostDist = []
    for ghost in currentGameState.getGhostStates():
        #if by the estimated time, ghost can come to pacman without being scared, then consider it a valid ghost
        if ghost.scaredTimer < manhattanDistance(pacPositon, ghost.getPosition()):
            ghostDist.append(manhattanDistance(pacPositon, ghost.getPosition()))
        #ignore the ghost since it will be scared
        else:
            ghostDist.append(float("inf"))
    #if sucessor is 0 away from ghost, do not go
    minGhostDist = min(ghostDist)
    if minGhostDist <= 0:
        return float("-inf")       
        
    #return the current score minus the estimated time to get to the closest food
    return currentGameState.getScore() - minFoodDist    

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

