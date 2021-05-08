import setuptools


setuptools.setup(
	name = "magma", 
	version = "0.0.1", 
	author = "Emanuel Flores", 
	author_email = "manuflores {at} caltech {dot} edu",
	description = "This repository contains neural network models based on pytorch to work with scRNAseq and small molecules.", 
	packages = setuptools.find_packages(), 
	classifiers = [
		"Programming Language :: Python :: 3", 
		"License :: OSI Approved :: MIT License", 
		"Operating System :: OS Independent"
	]
)
