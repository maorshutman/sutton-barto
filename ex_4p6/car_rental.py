"""
TODO
"""

import numpy as np

_GAMMA = 0.9
_MAX_CARS = 20
_THETA = 0.05
_A_MAX = 5


def policy_iteration(V, policy, gamma, A):
    old_policy = policy.copy()
    while True:
        V_new = eval_policy(V, policy, gamma, _THETA, A)
        new_policy, policy_stable = improve_policy(old_policy, V_new, gamma, A)

        print_policy(-new_policy)

        old_policy = new_policy.copy()

        if policy_stable:
            break

    return V_new, policy


def eval_policy(V, policy, gamma, theta, A):
    V_old = V.copy()
    V_new = V.copy()

    while True:
        delta = 0
        for n1 in range(_MAX_CARS):
            for n2 in range(_MAX_CARS):
                # print(n1, n2)
                V_new[n1, n2] = calc_mean_value(V_old, (n1, n2), policy[n1, n2], gamma)

                # print("v old", V_old[n1, n2])
                # print("v new", V_new[n1, n2])

                delta = max(delta, np.abs(V_new[n1, n2] - V_old[n1, n2]))

        V_old = V_new.copy()

        # print("v old", V_old[n1, n2])
        # print("v new", V_new[n1, n2])

        print("delta", delta)

        if delta < theta:
            break

    return V_new


def improve_policy(policy, V, gamma, A):
    new_policy = policy.copy()

    policy_stable = True
    for n1 in range(_MAX_CARS):
        for n2 in range(_MAX_CARS):
            old_action = policy[n1, n2]
            a = best_action((n1, n2), V, gamma, A)
            new_policy[n1, n2] = a

            if a != old_action:
                policy_stable = False

    return new_policy, policy_stable


def best_action(s, V, gamma, A):
    """Return pi(s)."""
    max_val = -1e10
    best_action = -10
    for a in A:
         # Do only legal actions A(s).
        if (s[0] + a >= 0) and (s[1] - a >= 0):
            v = calc_mean_value(V, s, a, gamma)
            if v > max_val:
                max_val = v
                best_action = a
    return best_action


def calc_mean_value(V, s, a, gamma):
    assert (s[0] + a >= 0) and (s[1] - a >= 0)

    v = 0.
    for n1 in range(_MAX_CARS):
        for n2 in range(_MAX_CARS):
            if (s[0] + a >= 0) and (s[1] - a >= 0):
                prob, exp_reward = p_env((n1, n2), s, a)
                v += exp_reward + gamma * V[n1, n2] * prob

    return v


def poisson(k, mu):
    return np.exp(-mu) * (mu**k) / np.math.factorial(k)


def p_env(s_next, s, a):
    """Given the state at the end of the previous day, calc the probability to
        have n1_next and n2_next at the end of the current day.

        s = (n1, n2), n1 <= 20, n2 <= 20
        a = number of moved cars from 1 -> 2.
    """
    mu1_rental = 3
    mu2_rental = 4
    mu1_return = 3
    mu2_return = 2

    # Before action - end of prev day.

    n1, n2 = s
    assert ((n1 <= _MAX_CARS) and (n1 >= 0))
    assert ((n2 <= _MAX_CARS) and (n2 >= 0))

    # After the action. We account for the fact that if there are too many cars
    # they disappear.
    assert ((n1 + a) >= 0) and ((n2 - a) >= 0)
    n1 += min(a, _MAX_CARS)
    n2 -= min(a, _MAX_CARS)

    # After action - next day begins.

    n1_next, n2_next = s_next
    assert ((n1_next <= _MAX_CARS) and (n1_next >= 0))
    assert ((n2_next <= _MAX_CARS) and (n2_next >= 0))

    # Init with reward due to moving cars.
    exp_reward = -2 * np.abs(a)

    p_n1_next = 0.
    for rental in range(0, _MAX_CARS + 1):
        actually_rented = min(rental, n1)
        required_ret = n1_next - n1 + actually_rented
        if required_ret >= 0:
            p_n1_next += poisson(rental, mu1_rental) *\
                         poisson(required_ret, mu1_return)
            exp_reward += p_n1_next * (10. * actually_rented)

    p_n2_next = 0.
    for rental in range(0, _MAX_CARS + 1):
        actually_rented = min(rental, n2)
        required_ret = n2_next - n2 + actually_rented
        if required_ret >= 0:
            p_n2_next += poisson(rental, mu2_rental) *\
                         poisson(required_ret, mu2_return)
            exp_reward += p_n2_next * (10. * actually_rented)

    assert (p_n1_next <= 1.) and (p_n2_next <= 1.)

    # As these are independent events we have a product.
    prob = p_n1_next * p_n2_next

    return prob, exp_reward


def print_values(V):
    print("V(n1, n2)")
    print(np.flip(V, axis=0))


def print_policy(policy):
    print("policy(n1, n2)")
    print(np.flip(policy, axis=0))


def main():
    # The set of actions.
    A = list(range(-_A_MAX, _A_MAX + 1))

    V = np.zeros((_MAX_CARS, _MAX_CARS))
    policy = np.zeros((_MAX_CARS, _MAX_CARS))

    policy_iteration(V, policy, _GAMMA, A)

    # Total prob should be 1.
    # tot = 0.
    # for i in range(_MAX_CARS + 1):
    #     for j in range(_MAX_CARS + 1):
    #         print ("state:", i, j)
    #         tot += p_env((i, j), (2, 18), 2)[0]
    # print(tot)


if __name__ == "__main__":
    main()
