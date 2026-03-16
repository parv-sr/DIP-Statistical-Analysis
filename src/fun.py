from abc import ABC, abstractmethod

class LandAnimal(ABC):
    def __init__(self, colour, **kwargs):
        super().__init__(**kwargs)
        self.colour = colour

    @abstractmethod
    def make_sound(self):                       #name hiding 
        print("Animal sounds I guess")


class Mammal(ABC):
    def __init__(self, has_offspring=True, **kwargs):
        super().__init__(**kwargs)
        self.has_offspring = has_offspring

    @abstractmethod    
    def child(self):
        print("Child")


class Horse(LandAnimal, Mammal):
    def __init__(self):
        super().__init__(colour="Brown", has_offspring=True)

    def make_sound(self):
        print("Neigh")

    def child(self):
        print("Foal")



class Dog(LandAnimal, Mammal):
    def __init__(self):
        super().__init__(colour="Black", has_offspring=True)

    def make_sound(self):
        print("Bark")
    
    def child(self):
        print("Pup")



class Cat(LandAnimal, Mammal):
    def __init__(self):
        super().__init__(colour="White", has_offspring=True)

    def make_sound(self):
        print("Meow")
    
    def child(self):
        print("Kitten")


#---------------------------------------------

h = Horse()
h.make_sound()
print(h.colour)
h.child()

print("="*40)

d = Dog()
d.make_sound()
print(d.colour)
d.child()

print("="*40)

c = Cat()
c.make_sound()
print(c.colour)
c.child()