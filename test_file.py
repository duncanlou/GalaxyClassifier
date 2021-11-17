import inspect



class A:
    def __init__(self):
        print("class A")
        self.kangzhi()

    def kangzhi(self):
        print("楼康志你好")
        deguo = inspect.stack()[0][3]
        faguo = inspect.stack()[1][3]
        yingguo = inspect.currentframe()
        eluosi = inspect.getfile(self.__class__)
        print(deguo)
        print(faguo)
        print(yingguo)
        print(eluosi)



a = A()