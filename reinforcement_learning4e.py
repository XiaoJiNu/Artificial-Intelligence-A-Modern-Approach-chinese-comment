"""Reinforcement Learning (Chapter 21)"""

import random
from collections import defaultdict

from mdp4e import MDP, policy_evaluation


# _________________________________________
# 21.2 Passive Reinforcement Learning 
# 21.2.1 Direct utility estimation


class PassiveDUEAgent:
    """
    Passive (non-learning) agent that uses direct utility estimation
    on a given MDP and policy.

    import sys
    from mdp import sequential_decision_environment
    north = (0, 1)
    south = (0,-1)
    west = (-1, 0)
    east = (1, 0)
    policy = {(0, 2): east, (1, 2): east, (2, 2): east, (3, 2): None, (0, 1): north, (2, 1): north,
              (3, 1): None, (0, 0): north, (1, 0): west, (2, 0): west, (3, 0): west,}
    agent = PassiveDUEAgent(policy, sequential_decision_environment)
    for i in range(200):
        run_single_trial(agent,sequential_decision_environment)
        agent.estimate_U()
    agent.U[(0, 0)] > 0.2
    True
    """

    def __init__(self, pi, mdp):
        self.pi = pi
        self.mdp = mdp
        self.U = {}
        self.s = None
        self.a = None
        self.s_history = []
        self.r_history = []
        self.init = mdp.init

    def __call__(self, percept):
        s1, r1 = percept
        self.s_history.append(s1)
        self.r_history.append(r1)
        ##
        ##
        if s1 in self.mdp.terminals:
            self.s = self.a = None
        else:
            self.s, self.a = s1, self.pi[s1]
        return self.a

    def estimate_U(self):
        # this function can be called only if the MDP has reached a terminal state
        # it will also reset the mdp history
        assert self.a is None, 'MDP is not in terminal state'
        assert len(self.s_history) == len(self.r_history)
        # calculating the utilities based on the current iteration
        U2 = {s: [] for s in set(self.s_history)}
        for i in range(len(self.s_history)):
            s = self.s_history[i]
            U2[s] += [sum(self.r_history[i:])]
        U2 = {k: sum(v) / max(len(v), 1) for k, v in U2.items()}
        # resetting history
        self.s_history, self.r_history = [], []
        # setting the new utilities to the average of the previous
        # iteration and this one
        for k in U2.keys():
            if k in self.U.keys():
                self.U[k] = (self.U[k] + U2[k]) / 2
            else:
                self.U[k] = U2[k]
        return self.U

    def update_state(self, percept):
        """To be overridden in most cases. The default case
        assumes the percept to be of type (state, reward)"""
        return percept


# 21.2.2 Adaptive dynamic programming


class PassiveADPAgent:
    """
    [Figure 21.2]
    Passive (non-learning) agent that uses adaptive dynamic programming
    on a given MDP and policy.

    import sys
    from mdp import sequential_decision_environment
    north = (0, 1)
    south = (0,-1)
    west = (-1, 0)
    east = (1, 0)
    policy = {(0, 2): east, (1, 2): east, (2, 2): east, (3, 2): None, (0, 1): north, (2, 1): north,
              (3, 1): None, (0, 0): north, (1, 0): west, (2, 0): west, (3, 0): west,}
    agent = PassiveADPAgent(policy, sequential_decision_environment)
    for i in range(100):
        run_single_trial(agent,sequential_decision_environment)

    agent.U[(0, 0)] > 0.2
    True
    agent.U[(0, 1)] > 0.2
    True
    """

    class ModelMDP(MDP):
        """Class for implementing modified Version of input MDP with
        an editable transition model P and a custom function T."""

        def __init__(self, init, actlist, terminals, gamma, states):
            super().__init__(init, actlist, terminals, states=states, gamma=gamma)
            nested_dict = lambda: defaultdict(nested_dict)
            # StackOverflow:whats-the-best-way-to-initialize-a-dict-of-dicts-in-python
            self.P = nested_dict()

        def T(self, s, a):
            """Return a list of tuples with probabilities for states
            based on the learnt model P."""
            return [(prob, res) for (res, prob) in self.P[(s, a)].items()]

    def __init__(self, pi, mdp):
        self.pi = pi
        self.mdp = PassiveADPAgent.ModelMDP(mdp.init, mdp.actlist,
                                            mdp.terminals, mdp.gamma, mdp.states)
        self.U = {}
        self.Nsa = defaultdict(int)
        self.Ns1_sa = defaultdict(int)
        self.s = None
        self.a = None
        self.visited = set()  # keeping track of visited states

    def __call__(self, percept):
        s1, r1 = percept
        mdp = self.mdp
        R, P, terminals, pi = mdp.reward, mdp.P, mdp.terminals, self.pi
        s, a, Nsa, Ns1_sa, U = self.s, self.a, self.Nsa, self.Ns1_sa, self.U

        if s1 not in self.visited:  # Reward is only known for visited state.
            U[s1] = R[s1] = r1
            self.visited.add(s1)
        if s is not None:
            Nsa[(s, a)] += 1
            Ns1_sa[(s1, s, a)] += 1
            # for each t such that Ns′|sa [t, s, a] is nonzero
            for t in [res for (res, state, act), freq in Ns1_sa.items()
                      if (state, act) == (s, a) and freq != 0]:
                P[(s, a)][t] = Ns1_sa[(t, s, a)] / Nsa[(s, a)]

        self.U = policy_evaluation(pi, U, mdp)
        ##
        ##
        self.Nsa, self.Ns1_sa = Nsa, Ns1_sa
        if s1 in terminals:
            self.s = self.a = None
        else:
            self.s, self.a = s1, self.pi[s1]
        return self.a

    def update_state(self, percept):
        """To be overridden in most cases. The default case
        assumes the percept to be of type (state, reward)."""
        return percept


# 21.2.3 Temporal-difference learning


class PassiveTDAgent:
    """
    [Figure 21.4]
    The abstract class for a Passive (non-learning) agent that uses
    temporal differences to learn utility estimates. Override update_state
    method to convert percept to state and reward. The mdp being provided
    should be an instance of a subclass of the MDP Class.

    import sys
    from mdp import sequential_decision_environment
    north = (0, 1)
    south = (0,-1)
    west = (-1, 0)
    east = (1, 0)
    policy = {(0, 2): east, (1, 2): east, (2, 2): east, (3, 2): None, (0, 1): north, (2, 1): north,
              (3, 1): None, (0, 0): north, (1, 0): west, (2, 0): west, (3, 0): west,}
    agent = PassiveTDAgent(policy, sequential_decision_environment, alpha=lambda n: 60./(59+n))
    for i in range(200):
        run_single_trial(agent,sequential_decision_environment)

    agent.U[(0, 0)] > 0.2
    True
    agent.U[(0, 1)] > 0.2
    True
    """

    def __init__(self, pi, mdp, alpha=None):

        self.pi = pi
        self.U = {s: 0. for s in mdp.states}
        self.Ns = {s: 0 for s in mdp.states}
        self.s = None
        self.a = None
        self.r = None
        self.gamma = mdp.gamma
        self.terminals = mdp.terminals

        if alpha:
            self.alpha = alpha
        else:
            self.alpha = lambda n: 1 / (1 + n)  # udacity video

    def __call__(self, percept):
        s1, r1 = self.update_state(percept)
        pi, U, Ns, s, r = self.pi, self.U, self.Ns, self.s, self.r
        alpha, gamma, terminals = self.alpha, self.gamma, self.terminals
        if not Ns[s1]:
            U[s1] = r1
        if s is not None:
            Ns[s] += 1
            # 此句是时序差分更新公式，用观测到的下一个状态的即时奖励r和估计的下一状态的效用U[s1]与当前状态的效用U[s]的差值来更新当前状态的效用U[s]
            U[s] += alpha(Ns[s]) * (r + gamma * U[s1] - U[s])
        if s1 in terminals:
            self.s = self.a = self.r = None
        else:
            self.s, self.a, self.r = s1, pi[s1], r1
        return self.a

    def update_state(self, percept):
        """To be overridden in most cases. The default case
        assumes the percept to be of type (state, reward)."""
        return percept


# __________________________________________
# 21.3. Active Reinforcement Learning
# 21.3.2 Learning an action-utility function


class QLearningAgent:
    """
    [Figure 21.8]
    An exploratory Q-learning agent. It avoids having to learn the transition
    model because the Q-value of a state can be related directly to those of
    its neighbors.

    import sys
    from mdp import sequential_decision_environment
    north = (0, 1)
    south = (0,-1)
    west = (-1, 0)
    east = (1, 0)
    policy = {(0, 2): east, (1, 2): east, (2, 2): east, (3, 2): None, (0, 1): north, (2, 1): north,
              (3, 1): None, (0, 0): north, (1, 0): west, (2, 0): west, (3, 0): west,}
    q_agent = QLearningAgent(sequential_decision_environment, Ne=5, Rplus=2, alpha=lambda n: 60./(59+n))
    for i in range(200):
        run_single_trial(q_agent,sequential_decision_environment)

    q_agent.Q[((0, 1), (0, 1))] >= -0.5
    True
    q_agent.Q[((1, 0), (0, -1))] <= 0.5
    True
    """

    def __init__(self, mdp, Ne, Rplus, alpha=None):
        # 初始化Q-learning代理
        self.gamma = mdp.gamma  # 折扣因子
        self.terminals = mdp.terminals  # 终止状态集合
        self.all_act = mdp.actlist  # 所有可能的动作列表
        self.Ne = Ne  # 探索函数中的迭代限制
        self.Rplus = Rplus  # 达到迭代限制前分配的大值
        self.Q = defaultdict(float)  # Q值表，默认为0
        self.Nsa = defaultdict(float)  # 状态-动作对访问次数
        self.s = None  # 当前状态
        self.a = None  # 当前动作
        self.r = None  # 当前奖励

        # 设置学习率alpha
        if alpha:
            self.alpha = alpha
        else:
            self.alpha = lambda n: 1. / (1 + n)  # udacity video

    def f(self, u, n):
        """
        探索函数（Exploration function）。
        这个函数实现了一种平衡探索（exploration）和利用（exploitation）的策略。

        参数:
        u: 当前状态-动作对的估计效用值
        n: 当前状态-动作对被访问的次数

        返回:
        如果访问次数小于阈值Ne，返回一个较大的固定值Rplus，鼓励探索；
        否则返回实际的估计效用值u，倾向于利用已知信息。

        思想解释:
        1. 探索-利用权衡：在强化学习中，智能体需要在探索新的、可能更好的选择和利用当前最优选择之间取得平衡。
        2. 乐观初始化：通过返回一个较大的Rplus值，鼓励智能体在初期多尝试不同的动作，这是一种"乐观面对不确定性"的策略。
        3. 渐进式转向利用：随着访问次数增加，函数会逐渐从返回Rplus转向返回实际估计值u，这反映了对已获得信息的信心增加。
        4. 简单而有效：这种机制简单，但在实践中非常有效，能够在学习初期促进广泛探索，后期则更多地利用学到的知识。
        5. 参数化控制：通过调整Ne和Rplus，可以灵活地控制探索程度，适应不同的问题特性。

        这个函数是Q-learning算法中平衡探索与利用的关键组成部分，有助于智能体在未知环境中更有效地学习最优策略。
        """
        if n < self.Ne:
            return self.Rplus  # 鼓励探索
        else:
            return u  # 倾向于利用

    def actions_in_state(self, state):
        """Return actions possible in given state.
        Useful for max and argmax."""
        # 返回给定状态下可能的动作
        if state in self.terminals:
            return [None]
        else:
            return self.all_act

    def __call__(self, percept):
        # __call__实现的事情：基于感知结果更新Q值，最后返回下一个动作

        # 更新状态和奖励
        s1, r1 = self.update_state(percept)
        # 获取Q值表、状态-动作访问次数、当前状态、动作和奖励
        Q, Nsa, s, a, r = self.Q, self.Nsa, self.s, self.a, self.r
        # 获取学习率、折扣因子和终止状态集
        alpha, gamma, terminals = self.alpha, self.gamma, self.terminals,
        # 获取获取可用动作的函数
        actions_in_state = self.actions_in_state

        if s in terminals:
            Q[s, None] = r1  # 更新终止状态的Q值
        if s is not None:
            Nsa[s, a] += 1  # 增加状态-动作对的访问次数
            # 更新Q值
            Q[s, a] += alpha(Nsa[s, a]) * (r + gamma * max(Q[s1, a1]
                                                           for a1 in actions_in_state(s1)) - Q[s, a])
        if s in terminals:
            self.s = self.a = self.r = None  # 重置状态、动作和奖励
        else:
            self.s, self.r = s1, r1  # 更新当前状态和奖励
            # 选择下一个动作（探索与利用的平衡），这里利用探索函数来得到动作。在被动学习算法中，通过策略来选择动作
            self.a = max(actions_in_state(s1), key=lambda a1: self.f(Q[s1, a1], Nsa[s1, a1]))
        return self.a

    def update_state(self, percept):
        """To be overridden in most cases. The default case
        assumes the percept to be of type (state, reward)."""
        return percept


def run_single_trial(agent_program, mdp):
    """Execute trial for given agent_program
    and mdp. mdp should be an instance of subclass
    of mdp.MDP 
    
    这个run_single_trial函数模拟了一个智能体(agent)在马尔可夫决策过程(MDP)环境中进行单次试验的过程。以下是代码的主要功能:
    函数接收两个参数:
    agent_program: 智能体的决策函数。
    mdp: 马尔可夫决策过程的实例。
    2. 内部定义了一个嵌套函数take_single_action:
    它模拟在给定状态下采取某个动作的概率性结果。
    使用基于MDP提供的转移概率的加权采样。
    试验的主循环:
    从MDP的初始状态开始。
    在每次迭代中:
    a. 获取当前状态的奖励。
    b. 创建一个感知(状态,奖励)元组。
    c. 根据这个感知询问智能体下一步的动作。
    d. 如果智能体返回None,试验结束。
    e. 否则,使用take_single_action模拟执行该动作。
    f. 更新当前状态。
    这个函数本质上模拟了强化学习中的智能体-环境交互循环,其中智能体观察状态和奖励,决定一个动作,然后环境响应一个新的状态和奖励。
    这个试验会一直持续,直到智能体决定停止(通过返回None作为下一个动作)。
"""

    def take_single_action(mdp, s, a):
        """
        选择在状态s下执行动作a的结果，返回一个状态。

        参数:
        mdp: 马尔可夫决策过程实例
        s: 当前状态
        a: 要执行的动作

        返回:
        state: 执行动作后的新状态
        """
        # 生成一个0到1之间的随机数
        x = random.uniform(0, 1)
        # 初始化累积概率
        cumulative_probability = 0.0
        # 遍历所有可能的转移概率和状态
        """
        一、此函数在做什么？
            此函数是在状态s下执行动作a的结果，返回一个状态，在模拟实际环境的交互，得到一个具体的状态。原理如下：
            1. mdp.T(s, a) 返回一个列表，包含 (概率, 下一状态) 的元组。这些表示从当前状态 s 执行动作 a 后可能到达的所有下一个状态及其对应的概率。
            2. x = random.uniform(0, 1) 生成一个 0 到 1 之间的随机数。
            3. 代码遍历所有可能的下一个状态：
            累加每个状态的概率。
            检查随机数 x 是否小于累积概率。
            4. "如果随机数小于累积概率，选择当前状态" 这一步是算法的核心：
            想象一个从 0 到 1 的数轴，被划分成若干段，每段的长度对应一个状态的概率。
            随机数 x 就像是在这个数轴上随机落下的一个点。
            当 x 小于累积概率时，意味着它落在了对应当前状态的概率区间内。
            
            这种方法确保了：
            概率较大的状态有更大的机会被选中。
            长期来看，每个状态被选中的频率会接近其真实概率。
            
            举例：如果有三个可能的下一状态，概率分别是 0.2, 0.3, 0.5：
            如果 x < 0.2，选择第一个状态
            如果 0.2 ≤ x < 0.5，选择第二个状态
            如果 x ≥ 0.5，选择第三个状态
            这样，我们就实现了按照给定概率随机选择下一个状态。
        
        二、 实际环境中，还需要take_single_action吗？
            不需要，实际环境中可以和真实环境直接交互，不需要模拟环境来得到下一个状态。

        """
        for probability_state in mdp.T(s, a):
            probability, state = probability_state
            # 累加概率
            cumulative_probability += probability
            # 如果随机数小于累积概率，选择当前状态
            if x < cumulative_probability:
                break
        # 返回选中的新状态
        return state

    current_state = mdp.init
    while True:
        # 获取当前状态的奖励
        current_reward = mdp.R(current_state)
        # 创建感知元组，包含当前状态和奖励
        percept = (current_state, current_reward)
        # 调用智能体程序，根据感知决定下一步动作
        next_action = agent_program(percept)
        # 如果智能体返回None，表示到达终止状态，退出循环
        if next_action is None:
            break
        # 执行选定的动作，得到执行动作后的下一个状态，此处在模拟环境的交互，得到一个具体的状态。
        current_state = take_single_action(mdp, current_state, next_action)
