using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using UnityEngine.UI;

public class Earth : MonoBehaviour
{
    public GameObject fragment;
    public int health;
    private int maxHealth;
    public AudioSource destroySound;
    public int hitByFireCount = 0;
    public bool isBurning = false;
    public int hitByWaterCount = 0;
    public Image healthBar;
    float burnDamageDelay = 1.5f;
    float timeLastBurnDamage;
    public AudioSource burnSound;

    void Start() {
        maxHealth = health;
    }

    void Update() {
        GetComponent<Animator>().SetBool("isBurning", isBurning);
        if(!isBurning && hitByFireCount > 3) {
            if(burnSound.isPlaying == true) {
                burnSound.Stop();
            }
            isBurning = true;
            timeLastBurnDamage = Time.time;
        }
        if(isBurning) {
            if(Time.time - timeLastBurnDamage > burnDamageDelay) {
                timeLastBurnDamage = Time.time;
                TakeDamage(2);
            }
            if(burnSound.isPlaying == false) {
                burnSound.Play();
            }
            if(hitByWaterCount > 60) {
                isBurning = false;
                burnSound.Stop();
                hitByFireCount = 0;
                hitByWaterCount = 0;
            }
            

        }

        
            
        
    }
    

    public void Explode() {
        //generate 5 rock fragments within this object's box collider, and randomize its initial rotation.
        for(int i = 0; i < 12; i++) {
            GameObject fragmentInstance = Instantiate(fragment, transform.position, Quaternion.identity);
            fragmentInstance.transform.rotation = Quaternion.Euler(0, 0, Random.Range(0, 360));
        }
        destroySound.Play();
        //set collider and sprite off

        StartCoroutine(Die());
    }

    public void TakeDamage(int damage) {
        health -= damage;
        healthBar.fillAmount = (float)health / (float)maxHealth;
        //die if health is less than or equal to 0
        if(health <= 0) {
            StartCoroutine(Die());
        }

    }

    IEnumerator Die() {
        GetComponent<Collider2D>().enabled = false;
        GetComponent<SpriteRenderer>().enabled = false;
        isBurning = false;
        if(burnSound.isPlaying == true) {
            burnSound.Stop();
        }
        //set healthbar sprite off
        healthBar.transform.parent.gameObject.SetActive(false);
        GameObject player = GameObject.Find("Player");
        if(player.GetComponent<Player>() == null) {
            player.GetComponent<PlayerRL>().rockCount--;
        }  
        else {
            player.GetComponent<Player>().rockCount--;
        }
        yield return new WaitForSeconds(1f);
        //find player gameobject
        Destroy(gameObject);
    }

}
