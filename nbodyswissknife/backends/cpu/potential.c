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
    Py_RETURN_NONE;
}

// method definitions. setting the method names and argument types and description (docstrings)
static PyMethodDef potential_cpu_methods[] = {
    //"PythonName"      C-function Name      Argument presentation    description
    {"potential",       potential,               METH_VARARGS,           "asdasdasd"},
    {NULL,              NULL,                    0,                       NULL}             // sentinal
};


#if PY_MAJOR_VERSION == 2

PyMODINIT_FUNC initpotential_cpu(void)
{
    PyObject *module;
    module = Py_InitModule("potential_cpu", potential_cpu_methods);
    if ( module == NULL )
        return;
}

#elif PY_MAJOR_VERSION == 3

static struct PyModuleDef potential_cpumodule = {
    PyModuleDef_HEAD_INIT,
    "potential_cpu",
    "module with interface functions for computing potentials",
    -1,
    potential_cpu_methods
};

PyMODINIT_FUNC
PyInit_potential_cpu(void)
{
  return PyModule_Create(&potential_cpumodule);
}

#endif

