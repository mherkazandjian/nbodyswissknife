/*
  define a python module that defines interface functions for computing potentials
 */
#include <stdio.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <omp.h>

double
_potential(const double *x,
           const double *y,
           const double *z,
           const double *m,
           const double b,
           const double G,
           const double *r_loc,
           const int n)
{
  double pot = 0.0;
  const double b2 = b*b;

#pragma omp parallel for shared(x, y, z, m, r_loc) reduction(+:pot)
  for( int i = 0; i < n; i++ )
  {
    const double _x = x[i] - r_loc[0];
    const double _y = y[i] - r_loc[1];
    const double _z = z[i] - r_loc[2];
    const double r = sqrt(_x*_x + _y*_y + _z*_z + b2);

    pot += m[i] / r;
  }

  return - G * pot;
}

static
PyObject* potential(PyObject* self, PyObject *args)
{
  PyObject *_x=NULL, *_y=NULL, *_z=NULL, *_m=NULL;
  PyArrayObject *x=NULL, *y=NULL, *z=NULL, *m=NULL;

  PyObject *_b=NULL;
  double b=0.0;

  PyObject *_G=NULL;
  double G=0.0;

  double pot=0.0;
  PyObject *retval = NULL;

  PyObject *_r_loc=NULL;
  PyArrayObject *r_loc=NULL;

  if (!PyArg_ParseTuple(args, "OOOOOOO", &_x, &_y, &_z, &_m, &_b, &_G, &_r_loc))
    return NULL;

  x = (PyArrayObject *)PyArray_FROM_OTF(_x, NPY_DOUBLE, NPY_IN_ARRAY);
  if (x == NULL) return NULL;

  y = (PyArrayObject*)PyArray_FROM_OTF(_y, NPY_DOUBLE, NPY_IN_ARRAY);
  if (y == NULL) goto fail;

  z = (PyArrayObject*)PyArray_FROM_OTF(_z, NPY_DOUBLE, NPY_IN_ARRAY);
  if (z == NULL) goto fail;

  m = (PyArrayObject*)PyArray_FROM_OTF(_m, NPY_DOUBLE, NPY_IN_ARRAY);
  if (m == NULL) goto fail;

  b = PyFloat_AsDouble(_b);
  G = PyFloat_AsDouble(_G);

  r_loc = (PyArrayObject*)PyArray_FROM_OTF(_r_loc, NPY_DOUBLE, NPY_IN_ARRAY);
  if (r_loc == NULL) goto fail;

  const int n = PyArray_SIZE(x);   // number of points

  pot = _potential(
      (double *)PyArray_DATA(x),
      (double *)PyArray_DATA(y),
      (double *)PyArray_DATA(z),
      (double *)PyArray_DATA(m),
      b,
      G,
      (double *)PyArray_DATA(r_loc),
      n
  );

  Py_DECREF(x);
  Py_DECREF(y);
  Py_DECREF(z);
  Py_DECREF(m);
  Py_DECREF(r_loc);

  retval = Py_BuildValue("d", pot);
  return retval;

 fail:
   Py_XDECREF(x);
   Py_XDECREF(y);
   Py_XDECREF(z);
   Py_XDECREF(m);
   Py_DECREF(r_loc);

  return NULL;
}

// method definitions for the module
static PyMethodDef potential_cpu_methods[] = {
    //"PythonName"      C-function Name      Argument presentation    description
    {"potential",       potential,           METH_VARARGS,           "wrapper around _potential()"},
    {NULL,              NULL,                0,                       NULL}             // sentinal
};



#if PY_MAJOR_VERSION == 2

PyMODINIT_FUNC
initpotential_cpu(void)
{
  PyObject *module;
  module = Py_InitModule("potential_cpu", potential_cpu_methods);

  import_array();

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
  PyObject *module;
  module = PyModule_Create(&potential_cpumodule);

  import_array();

  return module;
}

#endif

