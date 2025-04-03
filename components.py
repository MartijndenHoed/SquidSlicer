import numpy as np

class Component:
    def __init__(self,position, z_level, height, component_rules, pad_rules, hole_rules, cover_rules):
        self.position = position
        self.actual_z_level = 0
        self.parent_z_level = z_level
        self.slicing_activated = False
        self.height = height
        self.component_rules = component_rules
        self.pad_rules = pad_rules
        self.hole_rules = hole_rules
        self.cover_rules = cover_rules
        return


    def generate_component_layer_array(self,dims,resolution):
        arr = np.zeros((resolution[0], resolution[1]), dtype=np.uint8)
        mesh = np.mgrid[0:resolution[0], 0:resolution[1]]
        for rule_set in self.component_rules:
            rule_set_arr = np.ones((resolution[0], resolution[1]), dtype=np.uint8)
            for rule in rule_set:
                rule = rule.replace('x',f'{np.cos(self.angle)} * p - {np.sin(self.angle)} * q')
                rule = rule.replace('y', f'{np.sin(self.angle)} * p + {np.cos(self.angle)} * q')

                rule = rule.replace('p',
                                    f"(((mesh[0]/{resolution[0]})-{self.position[0]})*({dims[0][1] - dims[0][0]}))")
                rule = rule.replace('q',
                                    f"(((mesh[1]/{resolution[1]})-{self.position[1]})*({dims[1][1] - dims[1][0]}))")
                rule_set_arr *= eval(rule)
            arr = np.logical_or(arr, rule_set_arr)
        return arr

    def generate_pad_layer_array(self,dims,resolution):
        arr = np.zeros((resolution[0], resolution[1]), dtype=np.uint8)
        mesh = np.mgrid[0:resolution[0], 0:resolution[1]]
        for rule_set in self.pad_rules:
            rule_set_arr = np.ones((resolution[0], resolution[1]), dtype=np.uint8)
            for rule in rule_set:
                rule = rule.replace('x',f'{np.cos(self.angle)} * p - {np.sin(self.angle)} * q')
                rule = rule.replace('y', f'{np.sin(self.angle)} * p + {np.cos(self.angle)} * q')
                rule = rule.replace('p', f"(((mesh[0]/{resolution[0]})-{self.position[0]})*({dims[0][1] - dims[0][0]}))")
                rule = rule.replace('q', f"(((mesh[1]/{resolution[1]})-{self.position[1]})*({dims[1][1] - dims[1][0]}))")
                rule_set_arr *= eval(rule)
            arr = np.logical_or(arr,rule_set_arr)
        return arr

    def generate_hole_layer_array(self,z_level,dims,resolution):
        arr = np.zeros((resolution[0], resolution[1]), dtype=np.uint8)
        mesh = np.mgrid[0:resolution[0], 0:resolution[1]]
        for rule_set in self.hole_rules:
            rule_set_arr = np.ones((resolution[0], resolution[1]), dtype=np.uint8)
            for rule in rule_set:
                rule = rule.replace('x',f'{np.cos(self.angle)} * p - {np.sin(self.angle)} * q')
                rule = rule.replace('y', f'{np.sin(self.angle)} * p + {np.cos(self.angle)} * q')
                rule = rule.replace('p',
                                    f"(((mesh[0]/{resolution[0]})-{self.position[0]})*({dims[0][1] - dims[0][0]}))")
                rule = rule.replace('q',
                                    f"(((mesh[1]/{resolution[1]})-{self.position[1]})*({dims[1][1] - dims[1][0]}))")
                rule = rule.replace('z',
                                    f"{z_level}")
                rule_set_arr *= eval(rule)
            arr = np.logical_or(arr, rule_set_arr)
        return arr

    def generate_cover_layer_array(self):

        return

class Component_smd_0805(Component):

    def __init__(self,position,angle,z_level):
        self.position = position
        self.angle = angle

        self.name = self.display_name()
        self.height = 0.45
        self.component_rules = [["x>-1","x<1","y>-0.6","y<0.6"]]
        self.pad_rules = [["x>0.6","x<0.6+1","y>-0.6","y<0.6"],["x<-0.6","x>-0.6-1","y>-0.6","y<0.6"]]
        self.hole_rules = [["y>-0.6","y<0.6","(x-1)<z","(x+1)>-z"]]
        self.cover_rules = None

        super().__init__(self.position, z_level,self.height, self.component_rules, self.pad_rules, self.hole_rules, self.cover_rules)

    @staticmethod
    def display_name():
        return "0805 SMD component"

class Component_smd_0805_larger(Component):

    def __init__(self,position,angle,z_level):
        self.position = position
        self.angle = angle

        self.name = self.display_name()
        self.height = 0.6
        self.component_rules = [["x>-1","x<1","y>-0.6","y<0.6"]]
        self.pad_rules = [["x>0.6","x<0.6+1.4","y>-0.6","y<0.6"],["x<-0.6","x>-0.6-1.4","y>-0.6","y<0.6"]]
        self.hole_rules = [["y>-0.9","y<0.9","(x-1.0)<z","(x+1.0)>-z"]]
        self.cover_rules = None

        super().__init__(self.position, z_level,self.height, self.component_rules, self.pad_rules, self.hole_rules, self.cover_rules)

    @staticmethod
    def display_name():
        return "0805 SMD component (larger)"

class Component_smd_0805_largest(Component):

    def __init__(self, position, angle, z_level):
        self.position = position
        self.angle = angle

        self.name = self.display_name()
        self.height = 0.6
        self.component_rules = [["x>-1", "x<1", "y>-0.6", "y<0.6"]]
        self.pad_rules = [["x>0.6", "x<0.6+1.4", "y>-0.6", "y<0.6"], ["x<-0.6", "x>-0.6-1.4", "y>-0.6", "y<0.6"]]
        self.hole_rules = [["y>-1.0", "y<1.0", "(x-1.4)<z", "(x+1.4)>-z"]]
        self.cover_rules = None

        super().__init__(self.position, z_level, self.height, self.component_rules, self.pad_rules, self.hole_rules,
                         self.cover_rules)

    @staticmethod
    def display_name():
        return "0805 SMD component (largest)"


class Component_smd_0805_mega(Component):

    def __init__(self, position, angle, z_level):
        self.position = position
        self.angle = angle

        self.name = self.display_name()
        self.height = 0.6
        self.component_rules = [["x>-1", "x<1", "y>-0.6", "y<0.6"]]
        self.pad_rules = [["x>0.6", "x<0.6+1.4", "y>-0.6", "y<0.6"], ["x<-0.6", "x>-0.6-1.4", "y>-0.6", "y<0.6"]]
        self.hole_rules = [["y>-1.1", "y<1.1", "(x-1.4)<z", "(x+1.4)>-z"]]
        self.cover_rules = None

        super().__init__(self.position, z_level, self.height, self.component_rules, self.pad_rules, self.hole_rules,
                         self.cover_rules)

    @staticmethod
    def display_name():
        return "0805 SMD component (mega)"