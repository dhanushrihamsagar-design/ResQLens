def assess_risk(detections):
    """
    Analyze detected objects and assign a disaster-risk level.

    detections format:
    [
        {"class": "car", "confidence": 0.80},
        {"class": "person", "confidence": 0.90}
    ]
    """

    # Get detected class names
    names = [d["class"].lower() for d in detections]

    # Remove duplicate object names
    unique_names = set(names)

    # --------------------------------
    # HIGH RISK
    # --------------------------------

    if "person" in unique_names and "car" in unique_names:
        return {
            "level": "HIGH",
            "reason": (
                "A person and vehicle were detected in a potentially "
                "hazardous disaster environment."
            ),
            "action": (
                "Avoid entering the affected area. Do not attempt to "
                "drive through flooded roads and follow emergency guidance."
            )
        }

    if "person" in unique_names and "truck" in unique_names:
        return {
            "level": "HIGH",
            "reason": (
                "A person and large vehicle were detected in a potentially "
                "hazardous disaster environment."
            ),
            "action": (
                "Move to a safe location and avoid the affected roadway."
            )
        }

    # --------------------------------
    # MEDIUM RISK
    # --------------------------------

    if "car" in unique_names:
        return {
            "level": "MEDIUM",
            "reason": (
                "A vehicle was detected in the disaster scene. "
                "The surrounding road conditions may present a vehicle hazard."
            ),
            "action": (
                "Avoid driving through visibly flooded or obstructed roads."
            )
        }

    if "truck" in unique_names or "bus" in unique_names:
        return {
            "level": "MEDIUM",
            "reason": (
                "A large vehicle was detected in the disaster scene."
            ),
            "action": (
                "Avoid the affected roadway and follow local safety guidance."
            )
        }

    if "person" in unique_names:
        return {
            "level": "MEDIUM",
            "reason": (
                "A person was detected in the disaster scene."
            ),
            "action": (
                "Stay away from hazardous areas and move toward a safe location."
            )
        }

    # --------------------------------
    # LOW RISK
    # --------------------------------

    return {
        "level": "LOW",
        "reason": (
            "No high-risk object combination was detected."
        ),
        "action": (
            "Continue monitoring the situation and follow local guidance."
        )
    }