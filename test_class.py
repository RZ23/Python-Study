class Airport:
    def __init__(self, airport_code, airport_name,airport_locations):
        self.airport_code = airport_code
        self.airport_name = airport_name
        self.airport_locations = airport_locations
    def add_hub(self,hub):
        self.hub = hub
    def get_info(self):
        print("Airport infor: /n")
        print("The "+self.airport_code+": ")
        print("Name: "+self.airport_name+"/n Location: "+self.airport_locations)

class Chinese_Airport(Airport):
    def get_pinyin(self,pinyin):
        self.pinyin = pinyin

