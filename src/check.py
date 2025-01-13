try:
    # קוד שעלול לגרום לשגיאה
    result = some_function()
except Exception as e:
    print(f"An error occurred: {e}") # הדפסת השגיאה לקונסול
    # או שמירה לקובץ לוג:
    with open("error.log", "a") as f:
        f.write(f"An error occurred: {e}\n")