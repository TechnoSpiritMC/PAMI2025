import time


def LDA(start, angle, deviation):
    """
    :param start: ``int`` → The time when the measurements were taken.
    :param angle: ``int`` → The last measured angle.
    :param deviation: ``int`` → The deviation in degrees / seconds
    """

    currentDeviation = (time.time() - start) * deviation

    print("Current Deviation:", currentDeviation)

    currentAngle = angle - currentDeviation
    return currentAngle


def RDA(lastAngles):
    """
    :param lastAngles: ``list[int]`` → The last measured angles (The more, the better).
    """

    averageAngle = sum(lastAngles) / len(lastAngles)
    return averageAngle


class Buffer:
    def __init__(self, array):
        self.array = array

    @staticmethod
    def InitiateBuffer(length):
        """
        :param length: ``int`` → The length of the desired buffer.
        """

        return [0 for i in range(length)]

    def Slip(self, value):
        """
        :param value: ``any`` → Something to add to that buffer.
        """

        for i in range(len(self.array) - 1):
            self.array[i] = self.array[i + 1]
        self.array[0] = None
        self.array[-1] = value

        return self.array

    def getAverageValue(self):
        return sum(self.array) / len(self.array)

    def getMinValue(self):
        return min(self.array[0])

    def getMaxValue(self):
        return max(self.array[0])


def AreSimilar(List, Tolerance):
    Average = sum(List) / len(List)
    Similar = True

    for item in List:

        difference = abs(Average - item)
        if difference < Tolerance:
            pass
        else:
            Similar = False
            break

    return Similar


def DefineAlgorithm(Values):
    BASETOLERANCE = 5
    d = []

    for _ in range(len(Values) - 1):
        i = (len(Values) - 1) - _
        d.append(Values[i] - Values[i - 1])

    Average = sum(d) / len(d)
    Tolerance = BASETOLERANCE + abs(Average / BASETOLERANCE)

    if AreSimilar(d, Tolerance):
        return "LDA", Average

    else:
        return "RDA", None


if __name__ == "__main__":
    startTime = time.time()
    values = [i*10 for i in range(10000)]
    Algorithm, Opt = DefineAlgorithm(values)

    print(Algorithm, Opt)

    if Algorithm == "LDA":
        print(__name__, " Using linear drift algorythm.")
        Average = Opt

        for i in range(len(values)):
            print(LDA(startTime, values[i], Average))

    elif Algorithm == "RDA":
        print(__name__, " Using random drift algorythm.")
