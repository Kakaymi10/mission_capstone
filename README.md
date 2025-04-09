# Microvision

## To Run the Models:

1. **Navigate to the `models_quantizer` directory**:
    ```bash
    cd models_quantizer
    ```

2. **Install the required models**:
    Follow the instructions to download and set up any necessary models.

3. **Install dependencies**:
    Install the required Python dependencies from `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the model**:
    Execute the `app.py` script to run the model:
    ```bash
    python app.py
    ```

---

## To Run the App:

1. **Navigate to the `microvision` directory**:
    ```bash
    cd microvision
    ```

2. **Add your Supabase credentials**:
    Create a `.env` file in the `microvision` directory and add your Supabase credentials. The file should look like this:
    ```text
    SUPABASE_URL=your_supabase_url
    SUPABASE_KEY=your_supabase_key
    ```

3. **Run Flutter commands**:
    Follow the relevant Flutter commands to run and build the app. For example:
    ```bash
    flutter pub get
    flutter run
    ```

