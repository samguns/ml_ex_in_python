import matplotlib.pyplot as plt


def plotdata(x, y):
    plt.plot(x, y, 'rx', label='MarketSize')
    plt.xlabel('Profit in $10,000s')
    plt.ylabel('Population of City in 10,000s')
    plt.legend()
    plt.show()
