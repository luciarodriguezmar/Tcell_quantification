## USE OF THE ANALYSIS
The code allows the analysis of volume SEM datasets segmented with VAST Lite in order to extract morphological features in 2D and 3D of cells. These features include the area of different organelles and the shape, in terms of roundness in 2D or sphericity in 3D.
## HOW TO USE
### 1.	ENVIRONMENT 
To set up the environment for running the code, create a new environment in anaconda by using the 'tcell.yml' file. 
In case this doesn't work, manually import all packages necessary for the code using conda install or pip install (packages under pip in the yml file will most likely require pip installing) using the same version as the one written in the yml file.
### 2.	SEGMENTATION AND EXPORT
The cells of interest have to be segmented in Vast in a hierarchical way, so that the parent layer includes the whole cell surface. Each organelle category is segmented as a sublayer dependent from the parent ‘cell’. In the same way, the ‘heterochromatin’ layer is dependent from the ‘nucleus’ layer. In case of wanting to analyze the cells morphology in 3D, each cell has to be segmented in all sections. 
Once completed, to export the segmentation for the analysis, in Vast: File > Export, and export the data using the following conditions:
* Multi-tile stack
* Mip level: 0
*  Export segmentation as ".png"
*  8 bit/px
*  Collapsed labels as in current view
*  Export currently selected label (and select each segment layer for each organelle manually).
Export each segment layer individually one by one and store each in a different folder. The name of the individual files must contain the expression ".vsseg_export_s".
### 3.	ANALYSIS
For the analysis, you will require to have in the same directory as the jupyter notebooks, the files Cells.py and MGFeatures.py.
#### 3.1.	ANALYSIS IN 2D
After exporting the segmentation, we can either analyze the morphological properties of the cells in 2D or in 3D. In either case, the features have to extracted in 2D in the first place. Run the code ‘2D_tcell_analysis.ipynb’.
If running the code on a different dataset, check the specifications of tile size on the ‘Cells.py’ file, as instructed in ‘2D_tcell_analysis.ipynb’.
 
#### 3.2.	ANALYSIS IN 3D
After 3.1., run the ‘3D_tcell_analysis.ipynb’ using the excel file generated with the 2D features. 
After obtaining the features in 3D, this code also allows running PCA analysis and kmeans clustering, as well as statistical analysis and visualization of the data in napari.
### 4. OLD FILES
In case of wanting to re-run old files for the analysis of FoxP3+ cells, the excel files needed for this analysis are in the folder excel_files_final.
