import numpy as np
import matplotlib.pyplot as py
import matplotlib.patches as pat


def generation(h):
    # Create export file
    f = open("part1.txt", "w+")
    c = open("part1.csv", "w")

    u = 0.0  # distance from origin robot has to travel
    t = 0.0  # time
    k = 0  # samples
    a = -2.0
    n = 15 / h
    epoch_count = 0
    sample_count = n * h  # sample interval = n (no of steps) * h (0.01)
    x = [float] * (int(n) + 2)
    # x = [float] * int((((15 - t) / h) + 2))
    x[0] = 1
    _t = [0]
    _u = [2]

    while t <= 15:
        if t < 5:
            u = 2
            _u.append(float(2))
        if 5 < t <= 10:
            u = 1
            _u.append(float(1))
        if 10 < t <= 15:
            u = 3
            _u.append(float(3))

        x[k + 1] = x[k] + h * (a * x[k] + (2 * u))

        if h == 0.01:
            if int(k) % sample_count == 10:
                f.write("%s, %s, %s\n" % (t, x[k], u))
                c.write("%s, %s, %s\n" % (t, x[k], u))
        elif h == 0.1:
            f.write("%s, %s, %s\n" % (t, x[k], u))
            c.write("%s, %s, %s\n" % (t, x[k], u))
            #  print("%s, %s, %s\n" % (t, x[k], u))
        else:
            f.write("%s, %s, %s\n" % (t, x[k], u))
            c.write("%s, %s, %s\n" % (t, x[k], u))

        _t.append(t)
        t += h  # steps
        k += 1  # iterator
        #  sample +=1
    f.close()
    el_plotto(_t, x, _u)


def box_muller_random(h):  # part 2
    sd = 0.001  # standard deviation
    mean = 0.0
    it = 0  # Iteration

    k = 0
    h = h

    # Split data arrays
    d_time = []
    d_x = []
    d_u = []
    d_noise = []

    # read in the file
    with open("part1.txt") as f:
        for line in f:
            split = line.split(',')
            split = [x.strip() for x in split]
            # t,x,u
            d_time.append(float(split[0]))
            d_x.append(float(split[1]))
            d_u.append(float(split[2]))

    # create part2 export file
    file = open("part2.txt", "w+")
    c = open("part2.csv", "w+")

    for i in range(k, len(d_time)):
        z1 = np.random.uniform(0, 2 * np.pi)
        b = sd * np.sqrt(abs(-2 * np.log(np.random.uniform(low=0, high=1))))

        t = float(d_time[i])
        x = float(d_x[i])
        # xNoise = float(data2[i]) # might not need
        u = float(d_u[i])

        if it == 0:
            z2 = b * np.sin(z1) + mean
            xNoise = x + z2
            it = 1
            file.write("%f, %f, %f, %f\n" % (t, x, xNoise, u))
            c.write("%f, %f, %f, %f\n" % (t, x, xNoise, u))
            d_noise.append(xNoise)
        else:
            z3 = b * np.cos(z1) + mean
            xNoise = x + z3
            it = 0
            file.write("%f, %f, %f, %f\n" % (t, x, xNoise, u))
            c.write("%f, %f, %f, %f\n" % (t, x, xNoise, u))
            d_noise.append(xNoise)

    file.close()

    der_plÜtter(d_time, d_x, d_noise, d_u)


def perceptron(h):
    # load
    # open pt2
    file = open("part2.txt", "r")
    # create pt3
    pt3 = open("part3Per.txt", "w")

    X = []  # x with noise data
    Xn = []  # X normalised
    samples = []

    l = 0.01  # learning rate

    for line in file:
        split = line.split(',')
        split = [i.strip() for i in split]
        X.append(float(split[2]))
    file.close()

    weights = [np.random.uniform(low=-0.5, high=0.5) for x in range(len(X))]
    theta = np.random.uniform(low=-0.5, high=0.5)  # single uniformally random between -0.5 & 0.5
    outputs = [float] * len(X)

    max = np.max(X)
    min = np.min(X)

    # X[i] = x_min / (x_max - x_min)
    Xn = [(x - min) / (max - min) for x in X]


    # train
    # start with random weights
    threshold = 0.00001  # error squared threshold
    error_squared = 1.5  #
    count = 0  # amount of iterations

    while error_squared > threshold and count < 1000:  # while error_squared is not equal to zero
        for i in range(len(X)-1):  # for each sample
            w = [3]
            x = [3]

            if i == 0:
                w = [0, 0, weights[i]]
                x = [0, 0, X[i]]
            elif i == 1:
                w = [0, weights[i - 1], weights[i]]
                x = [0, X[i - 1], X[i]]
            else:
                x = [X[i], X[i - 1], X[i - 2]]
                w = [weights[i], weights[i - 1], weights[i - 2]]

            net_sum = np.dot(w, x) + theta  #((w.index(0) * x.index(0)) + (w.index(1) * x.index(1)) + (w.index(2) * x.index(2)))
            output = 1 if net_sum >= 0 else -1

            error = Xn[i + 1] - output

            if error is not 0:
                weights[i] += 1 * error * X[i + 1]

            error_squared += 0.5 * error ** 2
            outputs[i] = output

        error_squared = error_squared / len(X)
        count += 1

    name = "Perceptron"
    pt3.write("%s, %s, %s\n" % (X, outputs, Xn))
    plotter(X, outputs, Xn, name)


def sigmoid_func(h):
    # load
    # open pt2
    file = open("part2.txt", "r")
    # create pt3
    pt3 = open("part3Sig.txt", "w")

    X = []  # x with noise data
    Xn = []  # X normalised

    l = 0.5  # learning rate

    for line in file:
        split = line.split(',')
        split = [i.strip() for i in split]
        X.append(float(split[2]))
    file.close()

    weights = [np.random.uniform(low=-0.5, high=0.5) for x in range(len(X))]
    theta = np.random.uniform(low=-0.5, high=0.5)  # single uniformally random between -0.5 & 0.5
    outputs = [float] * len(X)
    max = np.max(X)
    min = np.min(X)

    # X[i] = x_min / (x_max - x_min)
    Xn = [(x - min) / (max - min) for x in X]

    # train
    # start with random weights
    threshold = 0.00001  # error squared threshold
    error_squared = 1.5  #
    count = 0  # amount of iterations

    while error_squared > threshold and count < 1000:  # while error_squared is not equal to zero
        for i in range(len(X) - 1):  # for each sample

            if i == 0:
                w = [0, 0, weights[i]]
                x = [0, 0, X[i]]
            elif i == 1:
                w = [0, weights[i - 1], weights[i]]
                x = [0, X[i - 1], X[i]]
            else:
                x = [X[i], X[i - 1], X[i - 2]]
                w = [weights[i], weights[i - 1], weights[i - 2]]

            net_sum = np.dot(w, x) + theta
            output = 1 / (1 + np.exp(-net_sum))

            error = Xn[i + 1] - output

            if error is not 0:
                weights[i] += l * error * X[i + 1]

            error_squared += 0.5 * error ** 2
            # end for
            outputs[i] = output

        error_squared = error_squared / len(X)
        count += 1
    name = "sigmoid"
    pt3.write("%s, %s, %s\n" % (X, outputs, Xn))
    plotter(X, outputs, Xn, name)


def plotter(data, data2, data3, name): #sigmoid & perc
    py.clf()
    py.gca()

    data2[-1] = data2[-2]
    xRange = np.arange(0, len(data), 1)  # start, length of data, step size

    py.title("Graph of %s" % name)

    py.plot(xRange, data) # input
    py.plot(xRange, data2) # output
    py.plot(xRange, data3) # Expected

    label1 = pat.Patch(color="blue", label="X")
    label2 = pat.Patch(color="green", label="Output")
    label3 = pat.Patch(color="orange", label="Xn")

    py.legend(handles=[label1, label2, label3], loc="lower left")
    py.savefig("graph_output_%s.png" % name)


def el_plotto(data, data2, data3): # part 1
    py.clf()
    py.gca()

    py.title("Graph for X with step size (h) of: %s" % h)

    py.plot(data, data2) # output
    py.plot(data, data3) # Expected

    label1 = pat.Patch(label="X", color="orange")
    label2 = pat.Patch(label="U")

    py.legend(handles=[label1, label2], loc="lower right")
    py.savefig("graph_output_pt1.png")


def der_plÜtter(data, data2, data3, data4): # part2
    py.clf()
    py.gca()

    py.title("Graph for X with Step Size of %s with Noise " % h)

    py.plot(data, data2) # input
    py.plot(data, data3) # input + noise
    py.plot(data, data4) # Expected

    label1 = pat.Patch(label="U", color="green")
    label2 = pat.Patch(label="xNoise", color="orange")
    label3 = pat.Patch(label="X")

    py.legend(handles=[label1, label2, label3], loc="lower left")
    py.savefig("graph_output_pt2.png", )


if __name__ == "__main__":
    h = 0.1  # step size
    generation(h)

    h = 0.01  # step size
    box_muller_random(h)

    perceptron(h)
    sigmoid_func(h)
