import pint

def setup_converter():
    """Initialize the unit registry"""
    return pint.UnitRegistry()

def convert_length(value, from_unit, to_unit, ureg):
    """Convert length using Pint module"""
    try:
        quantity = value * ureg(from_unit)
        result = quantity.to(to_unit)
        return result
    except pint.errors.DimensionalityError:
        return "Error: Invalid unit conversion"
    except pint.errors.UndefinedUnitError:
        return "Error: Invalid unit"

def convert_area(value, from_unit, to_unit, ureg):
    """Convert area using Pint module"""
    try:
        quantity = value * ureg(from_unit)
        result = quantity.to(to_unit)
        return result
    except pint.errors.DimensionalityError:
        return "Error: Invalid unit conversion"
    except pint.errors.UndefinedUnitError:
        return "Error: Invalid unit"

def convert_specific_surface_area(value, from_unit, to_unit, ureg):
    """Convert specific surface area using Pint module"""
    try:
        quantity = value * ureg(from_unit)
        result = quantity.to(to_unit)
        return result
    except pint.errors.DimensionalityError:
        return "Error: Invalid unit conversion"
    except pint.errors.UndefinedUnitError:
        return "Error: Invalid unit"

def main():
    # Initialize unit registry
    ureg = setup_converter()
    
    while True:
        print("\nUnit Converter (using Pint module)")
        print("\nSelect conversion type:")
        print("1. Length")
        print("2. Area")
        print("3. Specific Surface Area")
        
        try:
            conversion_type = input("\nEnter your choice (1, 2, or 3): ")
            
            if conversion_type == "1":
                print("\nAvailable length units examples:")
                print("\nMetric Units:")
                print("- picometer (pm)")
                print("- nanometer (nm)")
                print("- micrometer (um)")
                print("- millimeter (mm)")
                print("- centimeter (cm)")
                print("- decimeter (dm)")
                print("- meter (m)")
                print("- decameter (dam)")
                print("- hectometer (hm)")
                print("- kilometer (km)")
                print("- megameter (Mm)")
                
                print("\nImperial/US Units:")
                print("- inch (inch, in)")
                print("- foot (ft)")
                print("- yard (yard, yd)")
                print("- mile (mile, mi)")
                print("- thou (thou, mil)")
                print("- league (league)")
                
                print("\nNautical Units:")
                print("- nautical mile (nmi)")
                print("- fathom (fathom)")
                print("- cable length (cable_length)")
                
                print("\nAstronomical Units:")
                print("- astronomical unit (au)")
                print("- light year (ly)")
                print("- parsec (pc)")
                
                print("\nAtomic Units:")
                print("- angstrom (angstrom)")
                print("- bohr (bohr)")
                
                value = float(input("\nEnter the value to convert: "))
                from_unit = input("Enter the source unit: ")
                to_unit = input("Enter the target unit: ")
                
                result = convert_length(value, from_unit, to_unit, ureg)
                print(f"\nResult: {result}")
                
            elif conversion_type == "2":
                print("\nAvailable area units examples:")
                print("\nMetric Units:")
                print("- square picometer (picometer**2 or pm²)")
                print("- square nanometer (nanometer**2 or nm²)")
                print("- square micrometer (micrometer**2 or μm²)")
                print("- square millimeter (millimeter**2 or mm²)")
                print("- square centimeter (centimeter**2 or cm²)")
                print("- square decimeter (decimeter**2 or dm²)")
                print("- square meter (meter**2 or m²)")
                print("- square decameter (decameter**2 or dam²)")
                print("- square hectometer (hectometer**2 or hm²)")
                print("- square kilometer (kilometer**2 or km²)")
                print("- square megameter (megameter**2 or Mm²)")
                
                print("\nImperial/US Units:")
                print("- square inch (inch**2 or in²)")
                print("- square foot (foot**2 or ft²)")
                print("- square yard (yard**2 or yd²)")
                print("- square mile (mile**2 or mi²)")
                print("- square thou (thou**2)")
                print("- circular mil (cmil)")
                
                print("\nTraditional Units:")
                print("- acre")
                print("- hectare (ha)")
                print("- are (are)")
                print("- barn (barn)")
                print("- rood (rood)")
                
                print("\nSurvey Units:")
                print("- section")
                print("- township")
                print("- homestead")
                
                print("\nHistorical Units:")
                print("- arpent")
                print("- plaza")
                print("- varas castellanas")
                
                value = float(input("\nEnter the value to convert: "))
                from_unit = input("Enter the source unit: ")
                to_unit = input("Enter the target unit: ")
                
                result = convert_area(value, from_unit, to_unit, ureg)
                print(f"\nResult: {result}")
                
            elif conversion_type == "3":
                print("\nAvailable specific surface area units examples:")
                print("\nMetric Mass-Based Units:")
                print("- square meter per gram (meter**2/gram or m²/g)")
                print("- square meter per kilogram (meter**2/kilogram or m²/kg)")
                print("- square centimeter per gram (centimeter**2/gram or cm²/g)")
                print("- square millimeter per gram (millimeter**2/gram or mm²/g)")
                print("- square nanometer per gram (nanometer**2/gram or nm²/g)")
                print("- square kilometer per tonne (kilometer**2/tonne or km²/t)")
                
                print("\nImperial/US Mass-Based Units:")
                print("- square foot per pound (foot**2/pound or ft²/lb)")
                print("- square inch per pound (inch**2/pound or in²/lb)")
                print("- square yard per pound (yard**2/pound or yd²/lb)")
                print("- square mile per pound (mile**2/pound or mi²/lb)")
                print("- square foot per ounce (foot**2/ounce or ft²/oz)")
                print("- square inch per ounce (inch**2/ounce or in²/oz)")
                
                print("\nVolume-Based Units:")
                print("- square meter per cubic meter (meter**2/meter**3 or m²/m³)")
                print("- square foot per cubic foot (foot**2/foot**3 or ft²/ft³)")
                print("- square inch per cubic inch (inch**2/inch**3 or in²/in³)")
                
                print("\nMolar-Based Units:")
                print("- square meter per mole (meter**2/mole or m²/mol)")
                print("- square centimeter per mole (centimeter**2/mole or cm²/mol)")
                print("- square foot per mole (foot**2/mole or ft²/mol)")
                
                value = float(input("\nEnter the value to convert: "))
                from_unit = input("Enter the source unit: ")
                to_unit = input("Enter the target unit: ")
                
                result = convert_specific_surface_area(value, from_unit, to_unit, ureg)
                print(f"\nResult: {result}")
                
            else:
                print("Invalid choice. Please select 1, 2, or 3.")
                continue
                
        except ValueError:
            print("Error: Please enter a valid number")
        except Exception as e:
            print(f"Error: {str(e)}")
        
        if input("\nConvert another value? (y/n): ").lower() != 'y':
            break
    
    print("Thank you for using the converter!")

if __name__ == "__main__":
    main()