def AreSimilar(List, Tolerance):

    print(List)

    Average = sum(List) / len(List)
    Similar = True

    for item in List:

        difference = abs(Average - item)
        print(difference)

        if difference < Tolerance:
            pass
        else:
            Similar = False
            break

    return Similar


def DefineAlogorythm(Values):
    BASETOLERANCE = 5
    d = []

    for _ in range(len(Values)-1):
        i = (len(Values) - 1) - _

        print("i", Values[i], "i-1", Values[i-1])

        d.append(Values[i] - Values[i-1])

    Average = sum(d) / len(d)
    Tolerance = BASETOLERANCE + abs(Average/BASETOLERANCE)
    if AreSimilar(d, Tolerance):
        print(__name__, " Using linear drift algorythm.")

    else:
        print(__name__, "Using random drift algorythm.")


if __name__ == "__main__":
    values = [10, 20, 30, 40, 50]
    DefineAlogorythm(values)
