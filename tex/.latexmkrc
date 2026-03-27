# Compile in each source file's directory so relative imports resolve.
$do_cd = 1;

# Force PDF output even if the editor invokes latexmk without -pdf.
$pdf_mode = 1;

# Write PDFs alongside source TeX files; keep aux files in tex/out.
$out_dir = '';
$aux_dir = '../out';

# Make the shared tex/ directory available for class files and local assets.
$ENV{TEXINPUTS} = "..//:" . ($ENV{TEXINPUTS} // '');

# Disable SyncTeX sidecar files next to the PDFs.
$synctex = 0;

1;
