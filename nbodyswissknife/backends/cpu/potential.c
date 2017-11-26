/*
  define a python module that defines interface functions for computing potentials
 */
#include <stdio.h>
#include <Python.h>
#include <omp.h>

void _potential(double *m,
		double *x,
		double *y,
		double *z,
		double b,
		double *phi)
{


}

static PyObject* potential(PyObject* self, PyObject *args)
{
    printf("hello world\n");

    Py_RETURN_NONE;
}

// method definitions. setting the method names and argument types and description (docstrings)
static PyMethodDef potential_methods[] = {
    //"PythonName"      C-function Name      Argument presentation    description
    {"potential",       potential,               METH_VARARGS,           "asdasdasd"},
    {NULL,              NULL,                    0,                       NULL}             // sentinal
};


PyMODINIT_FUNC initpotentail(void)
{
    PyObject *module;
    module = Py_InitModule("potential", potential_methods);
    if ( module == NULL )
        return;
}
