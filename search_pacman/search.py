# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called 
by Pacman agents (in searchAgents.py).
"""

import util
from collections import defaultdict

class Node(object):
    ''' Represents a node path state chain that
    is used to make the generic search function
    generic.
    '''
    __slots__ = ['position', 'parent', 'action', 'cost', 'problem', 'depth']

    @staticmethod
    def create(problem):
        ''' Given an initial position, create a node
        :param problem: The initial problem to solve
        :returns: The initial starting node
        '''
        initial = problem.getStartState()
        return Node(problem, initial, None, "Stop", 0, 0)

    def __init__(self, problem, position, parent, action, cost, depth):
        ''' Initialize a new node instance
        :param problem: The problem state for the heuristic
        :param position: The position of this node
        :param parent: The previous parent node
        :param action: The action taken to get here
        :param cost: The total cost of the path to here
        :param depth: The total depth of the path to here
        '''
        self.position = position
        self.parent   = parent
        self.action   = action
        self.cost     = cost
        self.problem  = problem
        self.depth    = depth

    def append(self, state):
        ''' Given a new state, create a new path node
        :param state: The new state to append
        :returns: A new state node
        '''
        state, action, cost = state
        cost  = self.cost  + cost
        depth = self.depth + 1
        return Node(self.problem, state, self, action, cost, depth)

    def getPath(self):
        ''' Given a goal, return the path to that goal
        :returns: A path to the given goal
        '''
        path, node = [], self
        while node.parent != None:
            path.insert(0, node.action)
            node = node.parent
        return path

    def getPositions(self):
        ''' Given a goal, return the path of positions
        to that goal.
        :returns: A path of the positions to the given goal
        '''
        states, node = [], self
        while node.parent != None:
            states.insert(0, node.position)
            node = node.parent
        return states

    def contains(self, node):
        ''' Checks if the given state is already in this path
        :param state: The state to check for existence
        :returns: True if in this path, False otherwise
        '''
        # TODO make this O(1) with an instance singleton
        return node.position in self.getPositions()

    def __hash__(self):
        ''' An overload of the hash function
        :returns: A hash of the current state
        '''
        return hash(self.position)

    def __len__(self):
        ''' An overload of the len function
        :returns: The current length of the path
        '''
        return self.depth

    def __str__(self):
        ''' Returns a string representation of this node
        :returns: The representation of this node
        '''
        parent = str(self.parent.position) if self.parent else "(start)"
        params = (parent, self.action, self.cost, str(self.position), self.depth)
        return "node(%s %s(%d) %s) len(%d)" % params

class SearchProblem:
  """
  This class outlines the structure of a search problem, but doesn't implement
  any of the methods (in object-oriented terminology: an abstract class).
  
  You do not need to change anything in this class, ever.
  """
  
  def getStartState(self):
     """
     Returns the start state for the search problem 
     """
     util.raiseNotDefined()
    
  def isGoalState(self, state):
     """
       state: Search state
    
     Returns True if and only if the state is a valid goal state
     """
     util.raiseNotDefined()

  def getSuccessors(self, state):
     """
       state: Search state
     
     For a given state, this should return a list of triples, 
     (successor, action, stepCost), where 'successor' is a 
     successor to the current state, 'action' is the action
     required to get there, and 'stepCost' is the incremental 
     cost of expanding to that successor
     """
     util.raiseNotDefined()

  def getCostOfActions(self, actions):
     """
      actions: A list of actions to take
 
     This method returns the total cost of a particular sequence of actions.  The sequence must
     be composed of legal moves
     """
     util.raiseNotDefined()
           

def tinyMazeSearch(problem):
  """
  Returns a sequence of moves that solves tinyMaze.  For any other
  maze, the sequence of moves will be incorrect, so only use this for tinyMaze
  """
  from game import Directions
  s = Directions.SOUTH
  w = Directions.WEST
  return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
  """
  Search the deepest nodes in the search tree first
  [2nd Edition: p 75, 3rd Edition: p 87]

  Your search algorithm needs to return a list of actions that reaches
  the goal.  Make sure to implement a graph search algorithm
  [2nd Edition: Fig. 3.18, 3rd Edition: Fig 3.7].

  To get started, you might want to try some of these simple commands to
  understand the search problem that is being passed in:

  print "Start:", problem.getStartState()
  print "Is the start a goal?", problem.isGoalState(problem.getStartState())
  print "Start's successors:", problem.getSuccessors(problem.getStartState())
  """
  "*** YOUR CODE HERE ***"
  init_State = problem.getStartState()
  print init_State
  fringe = util.Stack()
  fringe.push( (init_State, [], []) )


  while not fringe.isEmpty():
        node, actions, visited = fringe.pop()
        print "Time for successors for {}".format(node)
        for state, direction, cost in problem.getSuccessors(node):
            print "state, direction, cost", state, direction, cost
            if not state in visited:
                if problem.isGoalState(state):
                    print 'Goal State', actions + [direction]
                    return actions + [direction]
                fringe.push((state, actions + [direction], visited + [node]))
                #print "Pushed stack", (state, actions + [direction], visited + [node])
  return []


def breadthFirstSearch(problem):
  """
  Search the shallowest nodes in the search tree first.
  [2nd Edition: p 73, 3rd Edition: p 82]
  """
  "*** YOUR CODE HERE ***"
  # init_state = problem.getStartState()
  #
  # queue = util.Queue()
  # queue.push((init_state, []))
  # visited = set()
  # frontierSet = set()
  #
  # while not queue.isEmpty():
  #     (node, path) = queue.pop()
  #     if problem.isGoalState(node):
  #         return path
  #     visited.add(node)
  #     successors = problem.getSuccessors(node)
  #     for state, action, cost in successors:
  #         if state not in visited and state not in frontierSet:
  #                 queue.push((state, path + [action]))
  #                 frontierSet.add(state)
  return genericSearch(problem,util.Queue())
      
def uniformCostSearch(problem):
  "Search the node of least total cost first. "
  "*** YOUR CODE HERE ***"
  cost = lambda node: node.cost
  return genericSearch(problem, util.PriorityQueueWithFunction(cost))

def genericSearch(problem, frontier):
    ''' a generic framework for searching
    :param problem: The problem to be solved
    :param frontier: The frontier structure to use
    :returns: A path to solve the specified problem
    '''
    start = Node.create(problem)
    visited = defaultdict(lambda: 0)
    frontier.push(start)

    while not frontier.isEmpty():
        state = frontier.pop()
        if problem.isGoalState(state.position):
            return state.getPath()

        visited["{}".format(state.position)] += 1
        if visited["{}".format(state.position)] > 1: continue
        for action in problem.getSuccessors(state.position):
            child = state.append(action)
            if "{}".format(child.position) not in visited:
                frontier.push(child)


def nullHeuristic(state, problem=None):
  """
  A heuristic function estimates the cost from the current state to the nearest
  goal in the provided SearchProblem.  This heuristic is trivial.
  """
  return 0

def aStarSearch(problem, heuristic=nullHeuristic):
  "Search the node that has the lowest combined cost and heuristic first."
  "*** YOUR CODE HERE ***"
  cost = lambda node : node.cost + heuristic(node.position,node.problem)
  return genericSearch(problem, util.PriorityQueueWithFunction(cost))

  
# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
