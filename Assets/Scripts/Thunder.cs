using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Thunder : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        StartCoroutine(Die());
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    IEnumerator Die() {
        yield return new WaitForSeconds(1.5f);
        Destroy(gameObject);
    }

    void OnTriggerEnter2D(Collider2D other) {
        if(other != null) {
            if(other.gameObject.tag == "Earth") {
                other.gameObject.GetComponent<Earth>().Explode();
            }
            if (other.gameObject.tag == "Enemy") {
                other.gameObject.GetComponent<Enemy>().TakeDamage(2.5f);
                //if the enemy isSlowed, call ChainLightning()
                if (other.gameObject.GetComponent<Enemy>().isSlowed) {
                    other.gameObject.GetComponent<Enemy>().ChainLightning();
                }

            }
        }
    }
}
