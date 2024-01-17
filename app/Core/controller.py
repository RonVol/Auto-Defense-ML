from Core.Attacks.EvasionAttacks import EvasionAttacks

class Controller:
    # all possible parameters user can input
    def __init__(self,classifier,x_test,y_test):
        self.classifier = classifier
        self.x_test = x_test
        self.y_test = y_test
    def run(self):
        evasion_att = EvasionAttacks(self.classifier,self.x_test,self.y_test)
        evasion_att.run()
        
