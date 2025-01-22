from pydantic import BaseModel

class PredictInput(BaseModel):
    Hydraulic_Pressure: float
    Coolant_Pressure: float
    Air_System_Pressure: float
    Coolant_Temperature: float
    Hydraulic_Oil_Temperature: float
    Spindle_Bearing_Temperature: float
    Spindle_Vibration: float
    Tool_Vibration: float
    Spindle_Speed: float
    Voltage: float
    Torque: float
    Cutting: float
