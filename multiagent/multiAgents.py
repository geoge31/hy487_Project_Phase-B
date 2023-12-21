# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
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
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

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

        #  Get Pacman's current position and the list of food coordinates

        currentPos = list(newPos)
        neg = -float("inf")  # Most negative value
        minDist = float("inf")  # Initialize minimum distance to food
        dist = 0  # Temporary variable for distance calculation
        currentFood = currentGameState.getFood()
        foodList = currentFood.asList()

        # Return the most negative value if the action is 'Stop'
        if action == "Stop":
            return neg

        # Calculate distances of foods and find the smallest distance
        for foodPosition in foodList:
            dist = manhattanDistance(foodPosition, currentPos)
            minDist = min(minDist, dist)

        # Reverse since it's the inverse of distance
        minDist = -minDist

        # If a ghost exists in the current position, return the most negative value
        for state in newGhostStates:
            if state.scaredTimer == 0 and state.getPosition() == tuple(currentPos):
                return neg

        return minDist


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

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def minimax(self, current_depth, agent_index, gameState):
        "Roll over agent index and increase current depth if all of the agents have finished playing their turn in a move"
        if agent_index >= gameState.getNumAgents():
            agent_index = 0
            current_depth += 1
        "Return the value of evaluationFunction if max depth is reached"
        if current_depth == self.depth:
            return None, self.evaluationFunction(gameState)
        " Best score and best action init"
        best_score = None
        best_action = None
        "Pacman turn"
        if agent_index == 0:
            for action in gameState.getLegalActions(
                agent_index
            ):  # For each legal action of pacman
                next_game_state = gameState.generateSuccessor(agent_index, action)
                temp, score = self.minimax(
                    current_depth, agent_index + 1, next_game_state
                )
                if best_score is None or score > best_score:
                    best_score = score
                    best_action = action
                "else: ghost turn"
        else:
            for action in gameState.getLegalActions(
                agent_index
            ):  # For each legal action of ghost agent
                next_game_state = gameState.generateSuccessor(agent_index, action)
                temp, score = self.minimax(
                    current_depth, agent_index + 1, next_game_state
                )
                if best_score is None or score < best_score:
                    best_score = score
                    best_action = action
        if best_score is None:
            return None, self.evaluationFunction(gameState)
        return best_action, best_score

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
        # util.raiseNotDefined()
        action, score = self.minimax(0, 0, gameState)
        return action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        inf = float("inf")
        action, score = self.alpha_beta(
            0, 0, gameState, -inf, inf
        )  # Get the action and score for pacman (max)
        return action  # Return the action to be done as per alpha-beta algorithm

    def alpha_beta(self, current_depth, agent_index, gameState, alpha, beta):
        if agent_index >= gameState.getNumAgents():
            agent_index = 0
            current_depth += 1
        if current_depth == self.depth:
            return None, self.evaluationFunction(gameState)
        best_score = None
        best_action = None
        if agent_index == 0:  # If it is max player's (pacman) turn
            for action in gameState.getLegalActions(
                agent_index
            ):  # For each legal action of pacman
                next_game_state = gameState.generateSuccessor(agent_index, action)
                temp, score = self.alpha_beta(
                    current_depth, agent_index + 1, next_game_state, alpha, beta
                )
                if best_score is None or score > best_score:
                    best_score = score
                    best_action = action
                alpha = max(alpha, score)
                if beta is not None and alpha is not None and alpha > beta:
                    break
        else:  # If it is min player's (ghost) turn
            for action in gameState.getLegalActions(
                agent_index
            ):  # For each legal action of ghost agent
                next_game_state = gameState.generateSuccessor(agent_index, action)
                temp, score = self.alpha_beta(
                    current_depth, agent_index + 1, next_game_state, alpha, beta
                )
                if best_score is None or score < best_score:
                    best_score = score
                    best_action = action
                beta = min(beta, score)
                if beta is not None and alpha is not None and beta < alpha:
                    break
        if best_score is None:
            return None, self.evaluationFunction(gameState)
        return best_action, best_score  # Return the best_action and best_score


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def expectimax(self, current_depth, agent_index, gameState):
        if agent_index >= gameState.getNumAgents():
            agent_index = 0
            current_depth += 1
        if current_depth == self.depth:
            return None, self.evaluationFunction(gameState)
        best_score = None
        best_action = None
        if agent_index == 0:
            for action in gameState.getLegalActions(agent_index):
                next_gameState = gameState.generateSuccessor(agent_index, action)
                temp, score = self.expectimax(
                    current_depth, agent_index + 1, next_gameState
                )
                if best_score is None or score > best_score:
                    best_score = score
                    best_action = action
        else:
            ghostActions = gameState.getLegalActions(agent_index)
            if len(ghostActions) is not 0:
                prob = 1.0 / len(ghostActions)
            for action in gameState.getLegalActions(agent_index):
                next_gameState = gameState.generateSuccessor(agent_index, action)
                temp, score = self.expectimax(
                    current_depth, agent_index + 1, next_gameState
                )
                if best_score is None:
                    best_score = 0.0
                best_score += prob * score
                best_action = action
            if best_score is None:
                return None, self.evaluationFunction(gameState)
            return best_action, best_score
        if best_score is None:
            return None, self.evaluationFunction(gameState)
        return best_action, best_score

    "Action function"

    def getAction(self, gameState):
        action, score = self.expectimax(0, 0, gameState)
        return action


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).
    DESCRIPTION: <write something here so we know what you did>
    Evaluate state by  :
          * closest food
          * food left
          * capsules left
          * distance to ghost
    """
    "*** YOUR CODE HERE ***"

    "Initiliazations and info from gameState fetching"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood().asList()
    extra = 0
    minFoodist = float("inf")
    ghostDistance = 0

    for ghost in currentGameState.getGhostPositions():
        ghostDistance = util.manhattanDistance(newPos, ghost)
        if ghostDistance < 2:
            return -float("inf")
    for food in newFood:
        minFoodist = min(minFoodist, util.manhattanDistance(newPos, food))

    foodLeft = currentGameState.getNumFood()
    capsLeft = len(currentGameState.getCapsules())

    foodLeftMultiplier = 950050
    capsLeftMultiplier = 10000
    foodDistMultiplier = 950

    if currentGameState.isLose():
        extra -= 50000
    elif currentGameState.isWin():
        extra += 50000

    return (
        1.0 / (foodLeft + 1) * foodLeftMultiplier
        + ghostDistance
        + 1.0 / (minFoodist + 1) * foodDistMultiplier
        + 1.0 / (capsLeft + 1) * capsLeftMultiplier
        + extra
    )


# Abbreviation
better = betterEvaluationFunction
