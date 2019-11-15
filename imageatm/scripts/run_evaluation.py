from imageatm.components import Evaluation


def run_evaluation(
	image_dir: str,
	job_dir: str,
	report: object,
	**kwargs,
):
    eval = Evaluation(image_dir=image_dir, job_dir=job_dir, **kwargs)
    eval.run(
		report_create = report['create'],
		report_kernel_name = report['kernel_name'],
		report_export_html = report['export_html'],
		report_export_pdf = report['export_pdf']
	)
