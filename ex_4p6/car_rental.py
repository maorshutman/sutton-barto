"""
Implementation of Jack' car rental problem from Sutton & Barto 2nd ed.
This implementation is naive and quite slow.
"""

import numpy as np
import matplotlib.pyplot as plt


_GAMMA = 0.9
_MAX_CARS = 20
_THETA = 0.1
_A_MAX = 5
_RENT_REWARD = 10.
_MOVE_REWARD = -2.
_ACTIONS = list(range(-_A_MAX, _A_MAX + 1))


def policy_iteration(V, policy):
    policy_old = policy.copy()
    V_old = V.copy()

    while True:
        V_new = eval_policy(V_old, policy_old)
        policy_new, policy_stable = improve_policy(policy_old, V_new)

        # DEBUG
        print_policy(policy_new)
        save_policy(policy_new)

        if policy_stable:
            break

        policy_old = policy_new.copy()
        V_old = V_new.copy()

    return V_new, policy_new


def eval_policy(V, policy):
    V_old = V.copy()
    V_new = np.zeros_like(V)

    while True:
        delta = 0
        for n1 in range(_MAX_CARS + 1):
            for n2 in range(_MAX_CARS + 1):
                a = policy[n1, n2]
                if legal_action((n1, n2), a):
                    V_new[n1, n2] = calc_mean_value(V_old, (n1, n2), a)
                    delta = max(delta, np.abs(V_new[n1, n2] - V_old[n1, n2]))

        V_old = V_new.copy()

        print("delta", delta)

        if delta < _THETA:
            break

    return V_new


def improve_policy(policy, V):
    new_policy = np.zeros_like(policy)

    policy_stable = True
    for n1 in range(_MAX_CARS + 1):
        for n2 in range(_MAX_CARS + 1):
            old_action = policy[n1, n2]

            a = best_action((n1, n2), V)
            new_policy[n1, n2] = a

            if int(a) != int(old_action):
                policy_stable = False

    return new_policy, policy_stable


def legal_action(s, a):
    return (s[0] - a >= 0) and (s[1] + a >= 0) and \
           (s[0] - a <= _MAX_CARS) and (s[1] + a <= _MAX_CARS)


def best_action(s, V):
    """Return pi(s)."""
    max_val = -1e10
    best_action = -_A_MAX - 1
    for a in _ACTIONS:
        if legal_action(s, a):
            v = calc_mean_value(V, s, a)
            if v > max_val:
                max_val = v
                best_action = a

    assert best_action != (-_A_MAX - 1)
    return best_action


def calc_mean_value(V, s, a):
    assert legal_action(s, a)

    v = 0.
    for n1 in range(_MAX_CARS + 1):
        for n2 in range(_MAX_CARS + 1):
            prob, exp_reward = p_env((n1, n2), s, a)
            v += exp_reward + _GAMMA * V[n1, n2] * prob

    return v


def poisson(k, mu):
    return np.exp(-mu) * (mu**k) / np.math.factorial(k)


def poisson_cum(min_k, mu):
    # TODO: Use the regularized gamma function.
    p = 0.
    for i in range(min_k, min_k + 10):
        p += poisson(i, mu)
    return p


def p_env(s_next, s, a):
    """Given the state at the end of the previous day, calc the probability to
        have n1_next and n2_next at the end of the current day, and calc the
        expected reward.

        In the books notation we mean:
        probability = SUM(r) p(s',r|s,a)
        expected reward = p(s'|s,a) = SUM(r) p(s',r|s,a) * r
    """
    mu1_rental = 3
    mu2_rental = 4
    mu1_return = 3
    mu2_return = 2

    # DEBUG
    # mu1_rental = 1e-10
    # mu2_rental = 1e-10
    # mu1_return = 1e-10
    # mu2_return = 1e-10

    # Before action - end of prev day.

    n1, n2 = s
    assert ((n1 <= _MAX_CARS) and (n1 >= 0))
    assert ((n2 <= _MAX_CARS) and (n2 >= 0))

    # After the action. We account for the fact that if there are too many cars
    # they disappear.
    assert legal_action((n1, n2), a)
    n1 = min(n1 - a, _MAX_CARS)
    n2 = min(n2 + a, _MAX_CARS)

    # After action - next day begins.

    n1_next, n2_next = s_next
    assert ((n1_next <= _MAX_CARS) and (n1_next >= 0))
    assert ((n2_next <= _MAX_CARS) and (n2_next >= 0))

    # Location 1 and 2 are independent, therefore we can do the following
    # decomposition:
    #   SUM (s', r) p(s',r|s,a)[r + g * V(s')] =
    # = SUM (n1', n2', r1, r2) p(n1',r|s,a) * p(n1',r|s,a) * [r1 + r2 + r(a) +
    #                                                         g * V(s')]

    # Location 1.
    p_n1_next = 0.
    exp_reward_1 = 0.
    for rental in range(0, _MAX_CARS + 1):
        actually_rented = min(rental, n1)
        required_ret = n1_next - n1 + actually_rented
        if required_ret >= 0:
            if n1_next == _MAX_CARS:
                p = poisson(rental, mu1_rental) * \
                    poisson_cum(int(required_ret), mu1_return)
            else:
                p = poisson(rental, mu1_rental) * \
                    poisson(required_ret, mu1_return)

            p_n1_next += p
            exp_reward_1 += p * actually_rented

    # Location 2.
    p_n2_next = 0.
    exp_reward_2 = 0.
    for rental in range(0, _MAX_CARS + 1):
        actually_rented = min(rental, n2)
        required_ret = n2_next - n2 + actually_rented
        if required_ret >= 0:
            if n2_next == _MAX_CARS:
                p = poisson(rental, mu2_rental) * \
                    poisson_cum(int(required_ret), mu2_return)
            else:
                p = poisson(rental, mu2_rental) * \
                    poisson(required_ret, mu2_return)

            p_n2_next += p
            exp_reward_2 += p * actually_rented

    exp_reward = _MOVE_REWARD * np.abs(a) * p_n1_next * p_n2_next + \
                 _RENT_REWARD * p_n1_next * exp_reward_2 + \
                 _RENT_REWARD * p_n2_next * exp_reward_1

    prob = p_n1_next * p_n2_next
    assert prob <= 1.

    return prob, exp_reward


def print_values(V):
    print("V(n1, n2)")
    print(np.flip(V, axis=0))


def print_policy(policy):
    print("policy(n1, n2)")
    print(np.flip(policy, axis=0))


def save_policy(policy):
    pi = np.flip(policy, axis=0)
    fig = plt.figure()
    plt.imshow(pi)
    plt.colorbar()
    fig.savefig("best_policy.png")


def main():
    V = np.zeros((_MAX_CARS + 1, _MAX_CARS + 1))
    policy = np.zeros((_MAX_CARS + 1, _MAX_CARS + 1))

    policy_iteration(V, policy)

    # DEBUG
    # tot = 0.
    # for i in range(_MAX_CARS + 1):
    #     for j in range(_MAX_CARS + 1):
    #         tot += p_env((i, j), (2, 5), 0)[0]
    # print(tot)


if __name__ == "__main__":
    main()
