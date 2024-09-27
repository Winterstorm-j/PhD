Creating tikz pictures
	- write standalone .tex file
	- activate python3.12 venv (may or may not need)
	`cd C:\Users\jstacey\Projects\.python3.12-env\Scripts && activate && cd "C:\Users\jstacey\OneDrive - ESR\WorkStuff\Projects\PhD"`
	- use pdflatex to compile tex to pdf
	`pdflatex Ch1pic.tex -synctex=1 -interaction=nonstopmode -shell-escape`
	- convert pdf to png with ImageMagick 7
	`magick -background white -density 600 Thesis\\Ch1pic.pdf Thesis\\assets\\simpleBN.png`
	- save png into assets directory
	- insert png into thesis 