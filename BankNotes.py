# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 19:37:10 2024

@author: Asus
"""

from pydantic import BaseModel
class BankNote(BaseModel):
    Distance: float
    ShipmentType: int
    NumPackages: int 
    SpecialHandling: int 
    PackageWeight: float 
    