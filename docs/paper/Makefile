
all: paper

clean:
	rm -f *.aux *.log *.dvi *.blg *.out *.toc

paper: High_resolution_imaging_requirements.pdf

# ----------------------------------------------------------------------------
# Paper I:

High_resolution_imaging_requirements.pdf: High_resolution_imaging_requirements.tex
	pdflatex High_resolution_imaging_requirements.tex
	pdflatex High_resolution_imaging_requirements.tex
	bibtex High_resolution_imaging_requirements
	pdflatex High_resolution_imaging_requirements.tex
	pdflatex High_resolution_imaging_requirements.tex
