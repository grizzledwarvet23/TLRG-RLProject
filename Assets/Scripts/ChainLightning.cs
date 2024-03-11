using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ChainLightning : MonoBehaviour
{

    public GameObject enemyOne, enemyTwo;

    void Start() {
        StartCoroutine(Die());
    }
    
    void Update()
    {
        //this gameobject has a linerenderer. set the first point to position of enemyOne, and the second point to position of enemyTwo. check for null
        if (enemyOne != null && enemyTwo != null) {
            GetComponent<LineRenderer>().SetPosition(0, enemyOne.transform.position);
            GetComponent<LineRenderer>().SetPosition(1, enemyTwo.transform.position);
        }
    }

    IEnumerator Die() {
        yield return new WaitForSeconds(0.8f);
        Destroy(gameObject);
    }
}
