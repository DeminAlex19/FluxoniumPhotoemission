import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy.optimize import root
from numpy import pi,linspace,tensordot
from scipy.sparse.linalg import eigsh
from scipy.integrate import solve_ivp
from scipy import interpolate
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FuncFormatter


def dagger(a):
    return np.conjugate(a.transpose())


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def funcHeatmap(x, pos):
    return "{:.2f}".format(x).replace("0.", ".").replace("1.00", "1")


def StatesRepr(state, bSize, size):
    # генерим подписи осей, где oX – первый базис, а oY – второй
    oX = np.zeros(size)
    oY = np.zeros(size)

    for n in range(size):
        oX[n] = str(n)
        oY[n] = str(n)

    # задаем матричку для диаграммы
    matrix = np.zeros((size, size))

    for n in range(size):
        for m in range(size):
            matrix[n, m] = abs(state[n + m * bSize]) ** 2

    fig, ax = plt.subplots()
    plt.rcParams["figure.figsize"] = (5, 5)
    im, _ = heatmap(matrix, oX, oY, ax=ax,
                    cmap="YlGn", vmin=0, vmax=1., cbarlabel="Probability")

    annotate_heatmap(im, valfmt=FuncFormatter(funcHeatmap), size=40 / np.sqrt(size))

    fig.tight_layout()
    plt.show()


def Oscillator(omega, numOfLevels=100):
    # собственные значения энергии
    eigEnergies = np.linspace(0, omega * (numOfLevels - 1), numOfLevels)

    # оператор уничтожения
    a = np.zeros((numOfLevels, numOfLevels), dtype=complex)
    for n in range(numOfLevels - 1):
        a[n, n + 1] = np.sqrt(n + 1)

    # оператор рождения
    at = np.zeros((numOfLevels, numOfLevels), dtype=complex)
    for n in range(numOfLevels - 1):
        at[n + 1, n] = np.sqrt(n + 1)

    return (eigEnergies, at, a)


def OperInEigStates(eigVectors, gridSize=0, h=0, leftBorder=0):
    # построим проекции канонических q и p на собственные векторы в q представлении

    # построим проектор pr (действ. на строки) и матрицы q и p в координатном базисе сетки (q)
    pr = eigVectors
    q = np.zeros((gridSize, gridSize), dtype=complex)
    p = np.zeros((gridSize, gridSize), dtype=complex)

    for n in range(gridSize):
        # поток
        q[n, n] = h * n + leftBorder

        # заряд
        if (n == 0):
            p[n, n + 1] = -1j / (2 * h)
        elif (n == gridSize - 1):
            p[n, n - 1] = 1j / (2 * h)
        else:
            p[n, n + 1] = -1j / (2 * h)
            p[n, n - 1] = 1j / (2 * h)

    # проецируем
    pNew = np.conjugate(pr.transpose()) @ p @ pr
    qNew = np.conjugate(pr.transpose()) @ q @ pr

    return (qNew, pNew)


def Fluxonium(Ej, El, Ec, gridSize=100, numOfLvls=100, leftBorder=-20, rightBorder=20, F=0):
    # Ej, El и Ec - эффективные энергии на джоз. эл., индуктивности и емкости

    # h - шаг сетки
    h = (rightBorder - leftBorder) / gridSize

    # H - матрица гамильтониана
    H = np.zeros((gridSize, gridSize), dtype=complex)

    # заполнение H по разностной схеме 2 порядка с нулевыми гран.усл.
    for n in range(gridSize):

        phi = h * n + leftBorder

        if (n == 0):
            H[n, n] = 2 * Ec / h ** 2 + El * (phi + 2 * np.pi * F) ** 2 - Ej * np.cos(phi - np.pi)
            H[n, n + 1] = -Ec / h ** 2
        elif (n == gridSize - 1):
            H[n, n] = 2 * Ec / h ** 2 + El * (phi + 2 * np.pi * F) ** 2 - Ej * np.cos(phi - np.pi)
            H[n, n - 1] = -Ec / h ** 2
        else:
            H[n, n] = 2 * Ec / h ** 2 + El * (phi + 2 * np.pi * F) ** 2 - Ej * np.cos(phi - np.pi)
            H[n, n - 1] = -Ec / h ** 2
            H[n, n + 1] = -Ec / h ** 2

    # диагонализация
    (eigEnergies, eigVectors) = eigsh(H, k=numOfLvls, which='SA', maxiter=4000)

    order = np.argsort(np.real(eigEnergies))
    eigEnergies = eigEnergies[order]
    eigVectors = eigVectors[:, order]

    (phi, q) = OperInEigStates(eigVectors, gridSize=gridSize, h=h, leftBorder=leftBorder)

    return (eigEnergies, phi, q)


def Transmon(Ej1, Ej2, Ec, gridSize=100, numOfLvls=100, F=0):
    # Ej и Ec - эффективные энергии на джоз. эл. и емкости

    # h - шаг сетки (из-за дескретности заряда шаг = 1)
    h = 1

    # H - матрица гамильтониана
    H = np.zeros((2 * gridSize + 1, 2 * gridSize + 1), dtype=complex)

    # заполнение H по разностной схеме 2 порядка с нулевыми гран.усл.
    for n in range(2 * gridSize + 1):

        q = h * n - gridSize

        if (n == 0):
            H[n, n] = Ec * q ** 2
            H[n, n + 1] = -(Ej1 + Ej2) / 2 * np.cos(F / 2) + (Ej2 - Ej1) / 2j * np.sin(F / 2)
        elif (n == 2 * gridSize):
            H[n, n] = Ec * q ** 2
            H[n, n - 1] = -(Ej1 + Ej2) / 2 * np.cos(F / 2) - (Ej2 - Ej1) / 2j * np.sin(F / 2)
        else:
            H[n, n] = Ec * q ** 2
            H[n, n - 1] = -(Ej1 + Ej2) / 2 * np.cos(F / 2) - (Ej2 - Ej1) / 2j * np.sin(F / 2)
            H[n, n + 1] = -(Ej1 + Ej2) / 2 * np.cos(F / 2) + (Ej2 - Ej1) / 2j * np.sin(F / 2)

    # диагонализация
    (eigEnergies, eigVectors) = eigsh(H, k=numOfLvls, which='SA', maxiter=4000)

    order = np.argsort(np.real(eigEnergies))
    eigEnergies = eigEnergies[order]
    eigVectors = eigVectors[:, order]

    (q, phi) = OperInEigStates(eigVectors, gridSize=2 * gridSize + 1, h=h, leftBorder=-gridSize)

    return (eigEnergies, phi, q)


def MixOfTwoSys(spect1, spect2, q1, q2, opers1=np.asarray([]), opers2=np.asarray([]), g=0, numOfLvls=5):
    # связываем две системы через операторы q1 и q2, попутно расширяя их операторы на общее пространство
    # opers – список из матриц операторов соотв. системы
    
    size1 = spect1.size
    size2 = spect2.size
    
    # единичная матрица 
    E1 = np.diag(np.linspace(1, 1, size1))
    E2 = np.diag(np.linspace(1, 1, size2))
    
    # диагонализованные гамильтонианы
    H1 = np.diag(spect1)
    H2 = np.diag(spect2)
    
    # объединяем линейные пространства
    H1 = np.kron(H1, E2)
    H2 = np.kron(E1, H2)    
    
    # q в общем базисе
    q1 = np.kron(q1, E2)
    q2 = np.kron(E1, q2)
    
    # полный гамильтониан
    H = H1 + H2 + g * q1@q2
                                   
    # диагонализация
    (eigEnergies, eigVectors) = eigsh(H, k=numOfLvls, which='SA', maxiter=4000)
    
    order=np.argsort(np.real(eigEnergies))
    eigEnergies=eigEnergies[order]
    eigVectors=eigVectors[:, order]
    
    # перетягиваем операторы
    if(opers1.shape[0] != 0):
        newOpers1 = np.zeros((opers1.shape[0], size1*size2, size1*size2), dtype=complex)
        for i in range(opers1.shape[0]):
            newOpers1[i, :, :] = np.kron(opers1[i, :, :], E2)
    
    if(opers2.shape[0] != 0):
        newOpers2 = np.zeros((opers2.shape[0], size1*size2, size1*size2), dtype=complex)
        for i in range(opers2.shape[0]):
            newOpers2[i, :, :] = np.kron(E1, opers2[i, :, :])
    
    if(opers1.shape[0] != 0 and opers2.shape[0] != 0):
        return (eigEnergies, eigVectors, H, newOpers1, newOpers2)
    elif(opers1.shape[0] != 0):
        return (eigEnergies, eigVectors, H, newOpers1)
    elif(opers2.shape[0] != 0):
        return (eigEnergies, eigVectors, H, newOpers2)
    else:
        return (eigEnergies, eigVectors, H)


def Graphs(t, X, x='x', y='y', full=False, save=False, filename=''):
    plt.rcParams["figure.figsize"] = (10, 10)

    for n in range(np.shape(X)[0]):
        lbl = str(n)
        plot = plt.plot(t, X[n, :], lw=1.5, label=lbl)

    plt.legend(loc='center left', bbox_to_anchor=(1.01, 0.5))

    if (full):
        Xf = np.zeros(np.shape(X)[1])
        for n in range(np.shape(X)[0]):
            for m in range(np.shape(X)[1]):
                Xf[m] += X[n, m]

        plot = plt.plot(t, Xf, color="gray", lw=1.5, label=lbl)

    # врубаем сетку
    plt.minorticks_on()

    # Определяем внешний вид линий основной сетки:
    plt.grid(which='major',
             color='k',
             linewidth=0.5)

    # Определяем внешний вид линий вспомогательной
    # сетки:
    plt.grid(which='minor',
             color='k',
             linestyle=':')

    plt.xlabel(x)
    plt.ylabel(y)

    if (save):
        plt.savefig(filename, facecolor='white')

    plt.show()


def PlotPcolormesh(fidelity, x, y, xlabel = 'x', ylabel = 'y', opt_lines=True, 
                    title=None, save=False, filename=''):
    fig, axs = plt.subplots(nrows = 1, ncols = 1,figsize = (10, 10))
    
    xGrid, yGrid = np.meshgrid(x, y)
    cmap_set = 'PiYG'
    cb = axs.pcolormesh(xGrid, yGrid, np.transpose(fidelity[:, :]), cmap = cmap_set)
    axs.set_xlabel(xlabel)
    axs.set_ylabel(ylabel)
    fig.colorbar(cb, ax=axs)

    opt_x_ind = np.argmax(np.real(fidelity))//fidelity.shape[1]
    opt_y_ind = np.argmax(np.real(fidelity))%fidelity.shape[1]
    
    
    axs.text(x[0], y[len(y)-1] + (y[len(y)-1] - y[0])*0.05, 
             'opt ' + ylabel + ' = ' + str(y[opt_y_ind]) + ' with index ' + str(opt_y_ind))
    axs.text(x[0], y[len(y)-1] + (y[len(y)-1] - y[0])*0.08, 
             'opt ' + xlabel + ' = ' + str(x[opt_x_ind]) + ' with index ' + str(opt_x_ind))
    axs.text(x[0], y[len(y)-1] + (y[len(y)-1] - y[0])*0.11, 
             'max fidelity = ' + str(np.abs(fidelity[opt_x_ind, opt_y_ind])))
    
    if opt_lines:
        axs.hlines(y[opt_y_ind], x[0], x[-1])
        axs.vlines(x[opt_x_ind], y[0], y[-1])
    if title != None:
        plt.title(title)

    if(save):
        plt.savefig(filename, facecolor = 'white')    
        
    plt.show()


def OperLin(A, B):
    # при линеаризации матрицы Ro по строкам в вектор ro преобразует A@Ro@B -> M@ro
    size = A.shape[0]
    M = np.zeros((size**2, size**2), dtype=complex) 
    for a in range(size):
        for b in range(size):
            for c in range(size):
                for d in range(size):
                    M[a*size + b, d + c*size] += A[a, c]*B[d, b]
                    
    return M


def PsiToRo(psi):
    # принимает psi - столбец!
    ro = psi@np.conjugate(psi.transpose())
    
    return ro


def LindbladLin(H, L):
    # возарвщвет эффективную матрицу M ур. Л. для лианеризованной по столбцам ro
    # L – массив операторов Линдблада
    size = H.shape[0]
    M = np.zeros((size**2, size**2), dtype=complex)
    
    # единичная матрица
    E = np.diag(np.linspace(1, 1, size))
    
    # бездиссипативная часть
    M += -1j*(OperLin(H, E) + OperLin(E, H))
    
    # диссипативная часть
    for n in range(L.shape[0]):
        M += OperLin(L[n], dagger(L[n])) +\
        1/2 * (OperLin(dagger(L[n])@L[n], E) +\
               OperLin(E, dagger(L[n])@L[n]))
        
    return M
