Pipeline resume
===============================================================================

The import pipeline has the potential to be a very long-running process, if 
you're importing large amounts of data, or performing complex transformations
on the data before writing.

While the pipeline runs, we take notes of our progress so that the pipeline can
be resumed at a later time, if the job is pre-empted or canceled for any reason.

Arguments
-------------------------------------------------------------------------------

When instantiating a pipeline, you can use the ``resume`` flag to indicate that
we can resume from an earlier execution of the pipeline.

If any resume files are found, we will only proceed if you've set the ``resume=True``.
Otherwise, the pipeline will terminate.

To address this, go to the temp directory you've specified and remove any intermediate
files created by the previous runs of the ``hipscat-import`` pipeline.