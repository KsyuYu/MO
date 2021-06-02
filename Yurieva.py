import math
from matplotlib import pyplot
print("----------------------------------------")
print("----------Khodos Zlata KM-62------------")
print("----------------------------------------")
print("min f(x1, x2) = 3*(x1-19)^2+x1*x2+4*x2^2")
print("----------------------------------------")

eps = 0.01


def f(x1, x2):
    return round(3 * ((x1 - 19) ** 2) + x1 * x2 + 4 * x2 ** 2, 4)


def norm(x):
    return round(math.sqrt((x[0]) ** 2 + (x[1]) ** 2), 4)


def grad(x1, x2):
    grad_1 = round((f(x1 + 0.01, x2) - f(x1, x2)) / 0.01, 4)
    grad_2 = round((f(x1, x2 + 0.01) - f(x1, x2)) / 0.01, 4)
    value_grad = [grad_1, grad_2]
    return value_grad


def norm_grad(value_grad):
    return round(math.sqrt(value_grad[0] ** 2 + value_grad[1] ** 2), 4)


x0 = [1.2 * 19 + 4, 1.2 * 19 + 4]


print("Start: x0 = ", x0)
print("Value in x0: ", f(x0[0], x0[1]))
print("Norm x0 ", norm(x0))

S1 = [1, 0]
S2 = [0, 1]
l = 0

print("Grad in x0 ", grad(x0[0], x0[1]), " and it`s norm", norm_grad(grad(x0[0], x0[1])))
print("-------------------------------------------")
print("Plan S2 - S1 - S2")



# x1=[x0[0], x0[1]+l*S2[1]]

def sven_search(x0):
    lambd = 0
    i = 0

    l = float(input("Enter lambda: "))

    f_minus = f(x0[0], x0[1] - l)
    f_zero = f(x0[0], x0[1])
    f_plus = f(x0[0], x0[1] + l)

    print('f_minus = ', f_minus, 'f_zero = ', f_zero, 'f_plus = ', f_plus)
        # print(x0[0], x0[1], i, l, f_minus)

    if f_minus <= f_zero:
        while f_minus < f_zero:
            lambd = round((lambd + (l * (2 ** i))), 5)
            print('lambd = ', lambd)

            f_zero = f(x0[0], x0[1])
            x0[1] = round((x0[1] - lambd), 5)
            f_minus = f(x0[0], x0[1])

            i += 1

            print('x0[0] = ', x0[0], 'x0[1] = ', x0[1], 'i = ', i, 'l = ', l, 'f_minus = ', f_minus, 'f_zero = ', f_zero)
            print("")

        x0[1] = x0[1] + lambd
        lambd = 0
        i = 0


def step_zper():
    print('\nmetod zolotoho pererizu')
    x1 = -23.5
    x4 = -7.5
    L = x4 - x1
    n = 0.001
    a = 1
    while a == 1:
        x2 = x1 + (L * 0.382)
        x3 = x4 - (L * 0.382)
        print('x1 = ', x1)
        print('x2 = ', x2)
        print('x3 = ', x3)
        print('x4 = ', x4)

        print('----------------')

        y1 = round(((3 * ((X1[0] + x1 - 19)**2)) + ((X1[0] + x1) * X1[1]) + (4 * (X1[1]**2))), 3)
        y2 = round(((3 * ((X1[0] + x2 - 19)**2)) + ((X1[0] + x2) * X1[1]) + (4 * (X1[1]**2))), 3)
        y3 = round(((3 * ((X1[0] + x3 - 19)**2)) + ((X1[0] + x3) * X1[1]) + (4 * (X1[1]**2))), 3)
        y4 = round(((3 * ((X1[0] + x4 - 19)**2)) + ((X1[0] + x4) * X1[1]) + (4 * (X1[1]**2))), 3)

        print('y1 = ', y1)
        print('y2 = ', y2)
        print('y3 = ', y3)
        print('y4 = ', y4)

        if y3 >= y2:
            x4 = x3
        elif y2 > y3:
            x1 = x2

        print('----------------')

        L = x4 - x1
        n1 = L / 4
        print('L = ', L)
        print('----------------')
        print('               ')
        if n1 < n:
            a = 0


def step_5t():
    print('metod 5 tochok')
    x1 = -23.5
    x5 = -7.5
    n = 0.01
    a = 1

    while a == 1:
        x3 = (x5 + x1) / 2
        x2 = (x3 + x1) / 2
        x4 = (x5 + x3) / 2
        print('x1 =', x1)
        print('x2 =', x2)
        print('x3 =', x3)
        print('x4 =', x4)
        print('x5 =', x5)

        print('----------------')

        y1 = round(((3 * ((x0[0] - 19)**2)) + (x0[0] * (x0[1] + x1)) + (4 * (x0[1] + x1)**2)), 3)
        y2 = round(((3 * ((x0[0] - 19)**2)) + (x0[0] * (x0[1] + x2)) + (4 * (x0[1] + x2)**2)), 3)
        y3 = round(((3 * ((x0[0] - 19)**2)) + (x0[0] * (x0[1] + x3)) + (4 * (x0[1] + x3)**2)), 3)
        y4 = round(((3 * ((x0[0] - 19)**2)) + (x0[0] * (x0[1] + x4)) + (4 * (x0[1] + x4)**2)), 3)
        y5 = round(((3 * ((x0[0] - 19)**2)) + (x0[0] * (x0[1] + x5)) + (4 * (x0[1] + x5)**2)), 3)

        print('y1 =', y1)
        print('y2 =', y2)
        print('y3 =', y3)
        print('y4 =', y4)
        print('y5 =', y5)

        if y2 <= y3 and y2 <= y4:
            x1 = x1
            x5 = x3
        elif y3 <= y2 and y3 <= y4:
            x1 = x2
            x5 = x4
        elif y4 <= y2 and y4 <= y3:
            x1 = x3
            x5 = x5
        print('----------------')

        L = x5 - x1
        n1 = L / 4
        print('L = ', L)
        print('----------------')
        print('               ')
        if n1 < n:
            a = 0

def step_dsk(X2):
    x1 = -23.5
    x2 = -15.5
    x3 = -7.5

    phi_1 = 3*(X2[0]-19)**2+X2[0]*(X2[1]+x1)+4*(X2[1]+x1)**2
    phi_2 = 3*(X2[0]-19)**2+X2[0]*(X2[1]+x2)+4*(X2[1]+x2)**2
    phi_3 = 3*(X2[0]-19)**2+X2[0]*(X2[1]+x3)+4*(X2[1]+x3)**2

    x_star = x2 + ((x2-x1)*(phi_1 - phi_3))/(2*(phi_1 - 2*phi_2 + phi_3))
    phi_star = 3*(X2[0]-19)**2+X2[0]*(X2[1]+x_star)+4*(X2[1]+x_star)**2
    print(phi_1, phi_2, phi_3, x_star, phi_star, sep=' ')

    x1 = x2
    x2 = x3
    x3 = x_star

    phi_1 = 3*(X2[0]-19)**2+X2[0]*(X2[1]+x1)+4*(X2[1]+x1)**2
    phi_2 = 3*(X2[0]-19)**2+X2[0]*(X2[1]+x2)+4*(X2[1]+x2)**2
    phi_3 = 3*(X2[0]-19)**2+X2[0]*(X2[1]+x3)+4*(X2[1]+x3)**2

    a1 = (phi_2 - phi_1)/(x2-x1)
    a2 = (1/(x3 - x2))*((phi_3 - phi_1)/(x3 - x1) - (phi_2 - phi_1)/(x2 - x1) )
    x_star = (x1+x2)/2 - a1/(2*a2)
    phi_star = 3*(X2[0]-19)**2+X2[0]*(X2[1]+x_star)+4*(X2[1]+x_star)**2
    print(phi_1, phi_2, phi_3, x_star, phi_star, sep=' ')
    print("Lambda with star = ", x_star)

#sven_search(x0)

print("=============S2 direction==================")
step_5t()
print('Please, enter minimum value X :')
min1 = round((float(input())), 3)
X1 = [x0[0], round((x0[1] + min1), 3)]
print('Grad[X1] = ', grad(X1[0], X1[1]), 'Norm grad = ', norm_grad(grad(X1[0], X1[1])))
print('X1 = ', X1)

print("=============S1 direction==================")
step_zper()
print('Please, enter minimum value X :')
min2 = round((float(input())), 3)
X2 = [X1[0] + min2, X1[1]]
print('Grad[X2] = ', grad(X2[0], X2[1]), 'Norm grad = ', norm_grad(grad(X2[0], X2[1])))
print('X2 = ', X2)

print("=============S2 direction==================")
step_dsk(X2)
print('Please, enter lambda with star :')
min3 = round((float(input())), 5)
X3 = [X2[0], X2[1]+min3]
print('Grad[X3] = ', grad(X3[0], X3[1]), 'Norm grad = ', norm_grad(grad(X3[0], X3[1])))
print('X3 = ', X3)

print("=============X3 - X1 direction==================")
S_new = [X3[0] - X1[0], X3[1] - X1[1]]
print('Please, enter the last step')
lam = round((float(input())), 4)
X4 = [X3[0] + lam * S_new[0], X3[1] + lam * S_new[1]]
print('Grad[X4] = ', grad(X4[0], X4[1]), 'Norm grad = ', norm_grad(grad(X4[0], X4[1])))
print('X4 = ', X4)


xlist = [x0[0], X1[0], X2[0], X3[0], X4[0]]
ylist = [x0[1], X1[1], X2[1], X3[1], X4[1]]
min_th = [19.404, -2.426]

pyplot.plot(xlist, ylist)

pyplot.scatter(min_th[0], min_th[1], color='red')

pyplot.show()

""" lam_default = 0.1 * (norm(X3)/norm(S_new))
print('Lambda by default = ', lam_default)
X4_default = [X3[0]+lam_default*S_new[0] , X3[1]+lam_default*S_new[1]]
print('Grad[X4_default] = ', grad(X4_default[0], X4_default[1]), 'Norm grad = ', norm_grad(grad(X4_default[0], X4_default[1])))
print('X4 by default = ', X4_default)
"""
