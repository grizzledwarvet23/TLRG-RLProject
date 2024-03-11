using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RockFragment : MonoBehaviour
{
    Rigidbody2D rb;
    public float speed;
    // Start is called before the first frame update
    void Start()
    {
        rb = GetComponent<Rigidbody2D>();
        StartCoroutine(Die());
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        rb.velocity  = transform.right * speed;
    }

    void OnTriggerEnter2D(Collider2D other) {
        if(other.gameObject.tag == "Enemy") {
            other.gameObject.GetComponent<Enemy>().TakeDamage(1);
            Destroy(gameObject);
        }
    }

    IEnumerator Die() {
        yield return new WaitForSeconds(3f);
        Destroy(gameObject);
    }
}
