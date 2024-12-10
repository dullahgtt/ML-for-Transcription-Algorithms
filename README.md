# Comparative Analysis of Speaker Diarization Techniques using Different Clustering Methods on CNN-Based Speaker Segmentation for Enhanced Precision and Recognition
### Developing a novel way to transcribe videos with accurate recognition of speakers.
---
This guide explains how to set up and run the MATLAB project for this research, including the workspace setup, adding necessary paths, retrieving data from an external link, and ensuring all required toolboxes are available.

---

### 1. Prerequisites

Before running the project, ensure you have the following installed:
- MATLAB (Recommended version: 2020b or later)
- Access to the internet for downloading external data from the ICSI corpus

---

### 2. Setting Up MATLAB Workspace

To ensure that your MATLAB workspace is ready, follow these steps:

1. **Open MATLAB**:
   - Launch the MATLAB application on your machine.
   
2. **Set the Current Folder**:
   - Navigate to the folder where the project files are located. This is your **working directory** for the MATLAB project.
   - You can do this either by using the **Current Folder** panel or by running the following command in the command window:
     ```matlab
     cd 'path_to_project_folder'
     ```
   - Replace `'path_to_project_folder'` with the actual directory path where your files are located.

---

### 3. Adding Paths

For each of the subfolders, you need to add these paths so MATLAB can access them. You can do this manually, or add the paths automatically using the following command:

1. **Adding paths manually**:
   - In MATLAB, navigate to the **Home tab** and click **Set Path**.
   - Add the directories where the scripts/functions are located by clicking on **Add Folder** or **Add with Subfolders** if there are nested folders.
   
2. **Automatically adding paths** (recommended for larger projects):
   - Create a script (e.g., `setup_paths.m`) to add the necessary paths to the MATLAB environment. The script can look like this:
     ```matlab
     % Example: Automatically add all folders and subfolders in the project directory
     addpath(genpath('path_to_project_folder'));
     ```
   - Run the script by typing `setup_paths` in the command window.

---

### 4. Retrieving Data

The project requires data from the ICSI corpus. Follow the steps below to download and retrieve the required data:

1. **Download Data from the ICSI Corpus**:
   - Visit the following link to access the data download page:
     [ICSI Data Download](https://groups.inf.ed.ac.uk/ami/icsi/download/#)
   - On the page, you will find the data from the ICSI corpus. Choose the datasets you need (in our case we downloaded all of the ICSI meetings), and follow the instructions to request access to the data. Make sure to read the data usage policies carefully.
   
2. **Download Data**:
   - After getting access, download the data manually from the website or use a tool like `wget` or `curl` if you have direct URLs.
   
   - **Note**: The data might come in a compressed format (e.g., `.zip` or `.tar.gz`), so you will need to extract it once downloaded.

3. **Extract Data** (if the data is compressed):
   - If the downloaded data is in a ZIP file or other compressed formats, use MATLAB’s `unzip` function to extract it:
     ```matlab
     unzip('data.zip', 'destination_folder');
     ```
   - Replace `'destination_folder'` with the folder where you want the data to be saved.

4. **Store Data in the Working Directory**:
   - Once extracted, store the data in a directory within your working folder to make sure the code can access it when running.

---

### 5. Required Toolboxes

The project relies on specific MATLAB toolboxes for functionality. Make sure the following toolboxes are installed:

1. **List of required toolboxes**:
   - Statistics and Machine Learning Toolbox
   - Optimization Toolbox
   - Image Processing Toolbox
   - MATLAB Compiler (if needed for deployment)

2. **Checking installed toolboxes**:
   - To verify which toolboxes are installed, run the following command in MATLAB:
     ```matlab
     ver
     ```
   - This will display a list of all installed toolboxes.

3. **Installing missing toolboxes**:
   - If any required toolbox is missing, you can install it via the **Add-Ons** menu in MATLAB or using the following command (requires administrative access):
     ```matlab
     matlab.addons.install('toolbox_name.mltbx');
     ```

---

### 6. Running the Code

Once the workspace is set up and all paths are added, you're ready to run the code.

1. **Run a Script**:
   - Locate the main script in your project (`SpeakerDiarization.m`).
   - In the MATLAB command window, simply type the script name (without the `.m` extension) and press **Enter**:
     ```matlab
     SpeakerDiarization
     ```

2. **Check for Output**:
   - Ensure that the output is saved or displayed correctly. If your script generates plots or logs, check the appropriate figures or files in the working directory.

---

By following these steps, you should be able to set up and run the research successfully in MATLAB. If you encounter any issues, feel free to contact myself or any other project collaborators or refer to the MATLAB documentation for further help.

