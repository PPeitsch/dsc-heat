# Determination of temperature and enthalpy of transformation in heat-flux DSC curves

**FILE STRUCTURE:**

Accepted data type: **ASCII**, encoding: **ANSI**
- The columns to be read are 5:
    1. Index (ºC)
    2. Time (s)
    3. Oven Temperature (ºC)
    4. Sample Temperature (ºC)
    5. Thermogravimetry (mg)
    6. Heat Flow (mW)

**NAME STRUCTURE:**
* Curve file to process: "filename.txt" # Spaces are allowed
* Instrumental curve file: "Blank filename.txt" # Spaces are allowed

**EXAMPLE:**
* Curve to process: "Cp Aluminum.txt"
* Instrumental curve: "Blank Cp Aluminum.txt"

**DETAILS AND RECOMMENDATIONS:**
1. It is recommended to use the same acquisition time for both the original curve and the instrumental curve.
2. The file paths where the files will be read from can be changed (experimental).
3. Make sure to only place files to be processed in the corresponding folders as a precaution.
4. The folders must already exist.